use std::sync::Arc;

use tokio::{io::AsyncReadExt, sync::mpsc};

pub mod ya {
    tonic::include_proto!("speechkit.stt.v3");
}

pub mod event {
    #[derive(Debug)]
    pub struct VendorMessage {
        pub timestamp: i64,
        pub is_final: bool,
        pub transcript: Option<String>,
    }
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    let oauth = std::env::var("YANDEX_OAUTH").unwrap();
    let folder_id = std::env::var("YANDEX_FOLDER").unwrap();
    println!("{} -> {:?}", folder_id, oauth);
    let audio = tokio::fs::read("audio.pcm").await.unwrap();
    println!("Loaded {} bytes", audio.len());

    let channel = tonic::transport::Channel::from_static("https://stt.api.cloud.yandex.net:443")
        .tls_config(tls_config())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = ya::recognizer_client::RecognizerClient::new(channel);
    let (streaming_tx, streaming_rx) = mpsc::channel(1024);
    let request_stream = tokio_stream::wrappers::ReceiverStream::new(streaming_rx);
    let mut input = tokio::fs::File::open("audio.pcm").await.unwrap();
    let silence_timeout = 1000;
    let mut timer = Some(0);
    let noinput_timeout = 5000;
    let alive = Arc::new(tokio::sync::Mutex::new(true));
    let alive2 = Arc::clone(&alive);

    let mut req = tonic::Request::new(request_stream);
    let iam_token = issue_iam_token(&oauth).await.unwrap();
    println!("IAM={:?}", iam_token);
    let meta = req.metadata_mut();
    meta.insert(
        "authorization",
        format!("Bearer {}", iam_token).parse().unwrap(),
    );
    meta.insert("x-folder-id", folder_id.parse().unwrap());

    let (vendor_tx, mut vendor_rx) =
        mpsc::channel::<Result<event::VendorMessage, tonic::Status>>(1024);
    tokio::spawn(async move {
        let mut recognized = None;
        while let Some(msg) = vendor_rx.recv().await {
            println!("FROM vendor: {:#?}", msg);
            match msg {
                Ok(e) => {
                    if e.is_final {
                        recognized = e.transcript;
                        break;
                    } else {
                        match e.transcript {
                            None => {
                                if let Some(started) = timer {
                                    if e.timestamp - started > noinput_timeout {
                                        println!("NOINPUT");
                                        break;
                                    }
                                }
                            }
                            partial => {
                                timer = None;
                                recognized = partial;
                            }
                        }
                    }
                }
                Err(status) => {
                    println!("FAILED with {:?}", status);
                    break;
                }
            }
        }
        if let Some(result) = recognized {
            println!("RECOGNIZED: {:?}", result)
        }
        *alive2.lock().await = false;
    });
    tokio::spawn(async move {
        let resp = client.recognize_streaming(req).await.unwrap();
        let mut stream = resp.into_inner();
        while let Ok(msg) = stream.message().await.inspect_err(|status| {
            println!("ERROR: {:?}", status);
            vendor_tx.blocking_send(Err(status.clone())).unwrap();
        }) {
            match msg {
                Some(ya::StreamingResponse {
                    audio_cursors:
                        Some(ya::AudioCursors {
                            partial_time_ms, ..
                        }),
                    event: Some(e),
                    ..
                }) => {
                    let vendor_msg = match e {
                        ya::streaming_response::Event::Partial(ya::AlternativeUpdate {
                            alternatives,
                            ..
                        }) => event::VendorMessage {
                            timestamp: partial_time_ms,
                            is_final: false,
                            transcript: alternatives.first().map(|a| a.text.clone()),
                        },
                        ya::streaming_response::Event::Final(ya::AlternativeUpdate {
                            alternatives,
                            ..
                        }) => event::VendorMessage {
                            timestamp: partial_time_ms,
                            is_final: false,
                            transcript: alternatives.first().map(|a| a.text.clone()),
                        },
                        ya::streaming_response::Event::EouUpdate(_eou_update) => {
                            println!("END of Uttrance");
                            continue;
                        }
                        ya::streaming_response::Event::FinalRefinement(ya::FinalRefinement {
                            r#type:
                                Some(ya::final_refinement::Type::NormalizedText(
                                    ya::AlternativeUpdate { alternatives, .. },
                                )),
                            ..
                        }) => event::VendorMessage {
                            timestamp: partial_time_ms,
                            is_final: true,
                            transcript: alternatives.first().map(|a| a.text.clone()),
                        },
                        _ => {
                            continue;
                        }
                    };
                    let _ = vendor_tx
                        .send(Ok(vendor_msg))
                        .await
                        .inspect_err(|_| println!("STILL receiving messages from STT"));
                }
                None => break,
                Some(resp) => {
                    println!("Unprocessed {:?}", resp);
                }
            }
        }
    });

    streaming_tx
    .send(ya::StreamingRequest {
        event: Some(ya::streaming_request::Event::SessionOptions(
            ya::StreamingOptions {
                recognition_model: Some(ya::RecognitionModelOptions {
                    model: "general".into(),
                    audio_format: Some(ya::AudioFormatOptions {
                        audio_format: Some(
                            ya::audio_format_options::AudioFormat::RawAudio(ya::RawAudio {
                                audio_encoding: ya::raw_audio::AudioEncoding::Linear16Pcm
                                    .into(),
                                sample_rate_hertz: 8000,
                                audio_channel_count: 1,
                            }),
                        ),
                    }),
                    text_normalization: Some(ya::TextNormalizationOptions {
                        text_normalization:
                            ya::text_normalization_options::TextNormalization::Enabled
                                .into(),
                        profanity_filter: false,
                        literature_text: true,
                        phone_formatting_mode:
                            ya::text_normalization_options::PhoneFormattingMode::Disabled
                                .into(),
                    }),
                    language_restriction: None,
                    audio_processing_type:
                        ya::recognition_model_options::AudioProcessingType::RealTime.into(),
                }),
                eou_classifier: Some(ya::EouClassifierOptions {
                    classifier: Some(
                        ya::eou_classifier_options::Classifier::DefaultClassifier(
                            ya::DefaultEouClassifier {
                                r#type: ya::default_eou_classifier::EouSensitivity::Default
                                    .into(),
                                max_pause_between_words_hint_ms: silence_timeout,
                            },
                        ),
                    ),
                }),
                recognition_classifier: None,
                speech_analysis: None,
                speaker_labeling: None,
            },
        )),
    })
    .await
    .unwrap();
    let mut buf = [0; 5120];
    let mut i = 0;
    let mut sent = 0;
    while let Ok(n) = input
        .read(&mut buf)
        .await
        .inspect_err(|io| println!("IO error: {:?}", io))
    {
        if n == 0 {
            break;
        }
        i += 1;
        streaming_tx
            .send(ya::StreamingRequest {
                event: Some(ya::streaming_request::Event::Chunk(ya::AudioChunk {
                    data: buf[..n].to_vec(),
                })),
            })
            .await
            .unwrap();
        sent += n;
        println!("{:4}. Have sent {} bytes.", i, sent);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        if !(*alive.lock().await) {
            break;
        }
    }
    println!("Sending stream is over");
}

fn tls_config() -> tonic::transport::ClientTlsConfig {
    tonic::transport::ClientTlsConfig::new().with_native_roots()
}

const IAM_TOKEN_KEY: &str = "iamToken";
const YANDEX_PASSPORT: &str = "yandexPassportOauthToken";

async fn issue_iam_token(oauth: &str) -> Option<String> {
    let client = reqwest::Client::new();
    let query = [(YANDEX_PASSPORT, oauth)];
    let req = client
        .post("https://iam.api.cloud.yandex.net/iam/v1/tokens")
        .query(&query);
    let resp = req
        .send()
        .await
        .inspect_err(|e| println!("[YA-ENGINE] Error sending request for IAM-token: {:?}", e))
        .expect("[YA-ENGINE] Send request to IAM-server fails.");
    let json: serde_json::Value = resp
        .json()
        .await
        .inspect_err(|e| {
            println!(
                "[YA-ENGINE] Response with IAM-token failed, JSON-error {:?}",
                e
            );
        })
        .unwrap();
    match &json[IAM_TOKEN_KEY] {
        serde_json::Value::String(s) => Some(s.to_owned()),
        _ => {
            println!(
                "[YA-ENGINE] No {:?} in response from token server {:?}",
                IAM_TOKEN_KEY, json
            );
            None
        }
    }
}
