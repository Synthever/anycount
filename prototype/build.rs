use std::{fs, path::Path};

fn main() {
    let dist = Path::new("web/dist");
    if !dist.exists() {
        let _ = fs::create_dir_all(dist);
        let _ = fs::write(
            dist.join("index.html"),
            "<!doctype html><html><body>AnyCount Prototype Dev Placeholder</body></html>",
        );
    }
    println!("cargo:rerun-if-changed=web/dist");
    println!("cargo:rerun-if-changed=migrations");
}
