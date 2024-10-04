pub mod ya {
    tonic::include_proto!("speechkit.stt.v3");
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
    let req = tokio_stream::iter(
        [
            ya::StreamingRequest {
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
                                        max_pause_between_words_hint_ms: 500,
                                    },
                                ),
                            ),
                        }),
                        recognition_classifier: None,
                        speech_analysis: None,
                        speaker_labeling: None,
                    },
                )),
            },
            ya::StreamingRequest {
                event: Some(ya::streaming_request::Event::Chunk(ya::AudioChunk {
                    data: audio,
                })),
            },
        ]
        .into_iter(),
    );
    let mut req = tonic::Request::new(req);
    let iam_token = issue_iam_token(&oauth).await.unwrap();
    println!("IAM={:?}", iam_token);
    let meta = req.metadata_mut();
    meta.insert(
        "authorization",
        format!("Bearer {}", iam_token).parse().unwrap(),
    );
    meta.insert("x-folder-id", folder_id.parse().unwrap());

    let resp = client.recognize_streaming(req).await.unwrap();
    let mut stream = resp.into_inner();
    while let Ok(msg) = stream
        .message()
        .await
        .inspect_err(|status| println!("ERROR: {:?}", status))
    {
        println!("Hello, world! {:#?}", msg);
        if msg.is_none() {
            break;
        }
    }
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
