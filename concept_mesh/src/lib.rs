use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyList};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
struct ConceptDiff {
    id: String,
    payload: String,
}

#[derive(Serialize, Deserialize)]
struct Concept {
    id: String,
    data: String,
}

/// Python class for Concept Mesh client
#[pyclass]
pub struct ConceptMesh {
    client: Client,
    base_url: String,
}

#[pymethods]
impl ConceptMesh {
    /// Primary constructor: accepts the mesh URL
    #[new]
    fn new(base_url: String) -> Self {
        ConceptMesh {
            client: Client::new(),
            base_url,
        }
    }

    /// Factory from a Python config dict
    #[staticmethod]
    fn from_config(config: &PyDict) -> PyResult<Self> {
        let url: String = config
            .get_item("url")
            .ok_or_else(|| PyRuntimeError::new_err("'url' key missing in config"))?
            .extract()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(ConceptMesh::new(url))
    }

    /// Send a concept diff
    fn record_diff(&self, diff: &PyDict) -> PyResult<()> {
        let id: String = diff
            .get_item("id")
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'id' in diff"))?
            .extract()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let payload: String = diff
            .get_item("payload")
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'payload' in diff"))?
            .extract()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let diff_model = ConceptDiff { id, payload };

        let resp = self
            .client
            .post(&format!("{}/record_diff", self.base_url))
            .json(&diff_model)
            .send()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(format!(
                "HTTP {} when recording diff",
                resp.status()
            )))
        }
    }

    /// Fetch all concepts
    fn get_concepts(&self, py: Python) -> PyResult<PyObject> {
        let resp = self
            .client
            .get(&format!("{}/concepts", self.base_url))
            .send()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(PyRuntimeError::new_err(format!(
                "HTTP {} when fetching concepts",
                resp.status()
            )));
        }

        let items: Vec<Concept> = resp
            .json()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let list = PyList::empty(py);
        for c in items {
            let dict = PyDict::new(py);
            dict.set_item("id", c.id)?;
            dict.set_item("data", c.data)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }
}

#[pymodule]
fn concept_mesh_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ConceptMesh>()?;
    Ok(())
}
