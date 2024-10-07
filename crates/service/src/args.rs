use argh::FromArgs;

#[derive(FromArgs, Debug)]
/// RaBitQ
pub struct Args {
    /// the RaBitQ saved directory
    #[argh(option, short = 'd')]
    pub dir: String,
    /// service port
    #[argh(option, short = 'p', default = "9000")]
    pub port: u32,
    /// S3 bucket
    #[argh(option, short = 'b')]
    pub bucket: String,
    /// S3 key prefix for the RaBitQ model data
    #[argh(option, short = 'k')]
    pub key: String,
    /// local cache directory
    #[argh(option, short = 'c')]
    pub cache_dir: String,
}
