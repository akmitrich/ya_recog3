fn main() {
    println!("cargo::rerun-if-changed=stt/v3/stt_service.proto");
    tonic_build::configure()
        .compile_protos(
            &["stt/v3/stt_service.proto", "stt/v3/stt.proto"],
            &[".", "googleapis"],
        )
        .unwrap();
}
