use genet_sdk::{prelude::*, writer::*};
use serde_derive::Deserialize;

use std::{
    fs::File,
    io::{BufWriter, Write},
};

#[derive(Deserialize)]
struct Arg {
    file: String,
}

#[derive(Clone)]
struct GenetFileWriter {}

impl Writer for GenetFileWriter {
    fn new_worker(&self, _ctx: &Context, arg: &str) -> Result<Box<Worker>> {
        let arg: Arg = serde_json::from_str(arg)?;
        let file = File::create(&arg.file)?;
        let writer = BufWriter::new(file);
        Ok(Box::new(GenetFileWorker {
            writer,
            entries: Vec::new(),
            layer_id: Token::null(),
        }))
    }

    fn metadata(&self) -> Metadata {
        Metadata {
            id: "app.genet.writer.genet-file".into(),
            filters: vec![FileType::new("genet", &["genet"])],
            ..Metadata::default()
        }
    }
}

struct GenetFileWorker {
    writer: BufWriter<File>,
    entries: Vec<genet_format::Entry>,
    layer_id: Token,
}

impl Worker for GenetFileWorker {
    fn write(&mut self, _index: u32, layer: &Layer) -> Result<()> {
        self.layer_id = layer.id();
        self.entries.push(genet_format::Entry {
            frame: genet_format::Frame {
                len: layer.data().len(),
            },
            data: layer.data(),
        });
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        let header = genet_format::Header {
            layer_id: self.layer_id.to_string(),
            entries: self.entries.len(),
        };
        let bin = bincode::serialize(&header)?;
        self.writer.write_all(&bincode::serialize(&bin.len())?)?;
        self.writer.write_all(&bin)?;
        for e in self.entries.iter() {
            let bin = bincode::serialize(&e.frame)?;
            self.writer.write_all(&bincode::serialize(&bin.len())?)?;
            self.writer.write_all(&bin)?;
            self.writer.write_all(&e.data)?;
        }
        Ok(())
    }
}

genet_writers!(GenetFileWriter {});
