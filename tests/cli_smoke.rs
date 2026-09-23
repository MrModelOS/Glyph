use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn glyphc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_glyphc"))
}

fn temp_root() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("glyphc-cli-test-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&path).expect("create CLI test directory");
    path
}

#[test]
fn nns_writes_output_file_and_check_is_quiet_on_stderr() {
    let root = temp_root();
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/nns/mlp.ns");
    let output = root.join("model.cpp");

    let result = glyphc()
        .args(["nns"])
        .arg(&input)
        .args(["--cpp", "--output"])
        .arg(&output)
        .output()
        .expect("run glyphc nns");
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.is_file());
    assert!(fs::read_to_string(&output).unwrap().contains("ns_matmul"));

    let check = glyphc()
        .args(["nns"])
        .arg(&input)
        .arg("--check")
        .output()
        .expect("run glyphc nns --check");
    assert!(check.status.success());
    assert_eq!(
        String::from_utf8_lossy(&check.stdout).trim(),
        "Shape checking passed."
    );
    assert!(
        check.stderr.is_empty(),
        "unexpected check diagnostics: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn new_preserves_absolute_paths_and_creates_scaffold() {
    let root = temp_root();
    let project = root.join("absolute-project");

    let result = glyphc()
        .args(["new"])
        .arg(&project)
        .output()
        .expect("run glyphc new");
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(project.join("glyph.toml").is_file());
    assert!(project.join("src/main.glyph").is_file());
    let manifest = fs::read_to_string(project.join("glyph.toml")).unwrap();
    assert!(manifest.contains("name = \"absolute-project\""));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn every_shipped_nns_example_passes_shape_check() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let examples = [
        root.join("examples/nns/mlp.ns"),
        root.join("examples/nns/moe_growth.ns"),
        root.join("examples/nns/transformer.ns"),
        root.join("examples/nns/dense.ns"),
        root.join("examples/nns/neumoe.ns"),
        root.join("examples/nns/static.ns"),
    ];
    for input in examples {
        let result = glyphc()
            .args(["nns"])
            .arg(&input)
            .arg("--check")
            .output()
            .expect("run glyphc nns --check");
        assert!(
            result.status.success(),
            "{} failed shape check: {}",
            input.display(),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn generic_codegen_is_reproducible() {
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/generics.glyph");
    let root = temp_root();
    let first = root.join("first.c");
    let second = root.join("second.c");
    for output in [&first, &second] {
        let result = glyphc()
            .args(["compile", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(output)
            .output()
            .expect("run glyphc compile");
        assert!(
            result.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn run_uses_a_fresh_temporary_directory() {
    // The generated reference C currently targets POSIX; keep the release
    // compiler testable on Windows until native Win32 headers are implemented.
    if cfg!(windows) || Command::new("gcc").arg("--version").output().is_err() {
        return;
    }
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/hello.glyph");
    let result = glyphc()
        .args(["run", "--input"])
        .arg(&input)
        .output()
        .expect("run glyphc run");
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("Hello"));
}
