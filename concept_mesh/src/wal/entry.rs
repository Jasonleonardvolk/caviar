//! stub

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    pub id: u64,
    pub data: Vec<u8>,
}