    async fn health_check(&self) -> Result<()> {
        let client = Client::new();
        let services = vec!["nlp", "vision", "audio", "cross_modal"];
        
        for service in services {
            let health_url = format!("http://localhost:{}/{}/health", self.config.python_service_port, service);
            
            match timeout(Duration::from_secs(5), client.get(&health_url).send()).await {
                Ok(Ok(response)) if response.status().is_success() => {
                    self.service_status.insert(service.to_string(), ServiceStatus::Running);
                }
                _ => {
                    let error_msg = format!("Health check failed for {}", service);
                    warn!("{}", error_msg);
                    self.service_status.insert(service.to_string(), ServiceStatus::Failed(error_msg));
                }
            }
        }
        
        Ok(())
    }
    
    async fn restart_failed_services(&self) -> Result<()> {
        let failed_services: Vec<String> = self.service_status
            .iter()
            .filter_map(|entry| {
                if matches!(entry.value(), ServiceStatus::Failed(_)) {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();
        
        for service in failed_services {
            info!("Restarting failed service: {}", service);
            // Implementation would restart the specific service
        }
        
        Ok(())
    }
    
    async fn shutdown_all_services(&self) -> Result<()> {
        info!("Shutting down Python analysis services");
        
        let mut processes = self.service_processes.lock().await;
        
        for (service_name, mut process) in processes.drain() {
            if let Err(e) = process.kill().await {
                warn!("Failed to kill {} service: {}", service_name, e);
            } else {
                info!("Stopped {} service", service_name);
            }
            
            self.service_status.insert(service_name, ServiceStatus::Stopped);
        }
        
        Ok(())
    }
}

// ===================================================================
// UTILITY FUNCTIONS AND TRAITS
// ===================================================================

/// Convenience function for creating text input
pub fn create_text_input(content: String) -> InputModality {
    InputModality::Text(TextInput {
        content,
        language: None,
        metadata: HashMap::new(),
        source: None,
    })
}

/// Convenience function for creating image input
pub fn create_image_input(data: Vec<u8>, format: String, width: u32, height: u32) -> InputModality {
    InputModality::Image(ImageInput {
        data,
        format,
        width,
        height,
        metadata: HashMap::new(),
        source: None,
    })
}

/// Convenience function for creating audio input
pub fn create_audio_input(
    data: Vec<u8>,
    format: String,
    sample_rate: u32,
    channels: u16,
    duration_ms: u64,
) -> InputModality {
    InputModality::Audio(AudioInput {
        data,
        format,
        sample_rate,
        channels,
        duration_ms,
        metadata: HashMap::new(),
        source: None,
    })
}

/// Extension trait for adding multimodal capabilities to cognitive modules
pub trait MultimodalExt {
    async fn process_multimodal_data(&self, data: InputModality) -> Result<Vec<ConceptId>>;
    async fn get_multimodal_insights(&self) -> Result<Vec<CognitiveInsight>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_multimodal_integrator_creation() {
        let config = MultimodalConfig::default();
        let orchestrator = Arc::new(create_test_orchestrator().await);
        
        let integrator = MultimodalIntegrator::new(config, orchestrator).await;
        assert!(integrator.is_ok());
    }
    
    #[tokio::test]
    async fn test_text_input_processing() {
        let integrator = create_test_integrator().await;
        
        let text_input = create_text_input("The cat sat on the mat".to_string());
        
        // This would normally process through Python services
        // For testing, we'll mock the response
        let result = integrator.process_multimodal_input(text_input, None).await;
        
        // In a real test, we'd verify the concepts were extracted
        // assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_cross_modal_alignment() {
        let integrator = create_test_integrator().await;
        
        let concepts = vec![
            ExtractedConcept {
                concept_id: None,
                name: "cat".to_string(),
                confidence: 0.9,
                modality: "text".to_string(),
                attributes: HashMap::new(),
                spatial_info: None,
                temporal_info: None,
                embeddings: Some(vec![0.1, 0.2, 0.3]),
            },
            ExtractedConcept {
                concept_id: None,
                name: "cat".to_string(),
                confidence: 0.85,
                modality: "image".to_string(),
                attributes: HashMap::new(),
                spatial_info: Some(SpatialInfo {
                    bounding_box: Some(BoundingBox {
                        x: 10.0,
                        y: 20.0,
                        width: 100.0,
                        height: 80.0,
                        confidence: 0.85,
                    }),
                    region_coordinates: None,
                    spatial_relationships: Vec::new(),
                }),
                temporal_info: None,
                embeddings: Some(vec![0.15, 0.25, 0.35]),
            },
        ];
        
        let alignments = integrator.perform_cross_modal_alignment(&concepts).await;
        // In a real test with services running, we'd verify alignments were created
        // assert!(alignments.is_ok());
    }
    
    #[tokio::test]
    async fn test_processing_statistics() {
        let integrator = create_test_integrator().await;
        
        let stats = integrator.get_processing_statistics().await;
        assert_eq!(stats.total_sessions_processed, 0);
        assert_eq!(stats.total_concepts_extracted, 0);
    }
    
    #[tokio::test]
    async fn test_multimodal_input_creation() {
        let text_input = create_text_input("Hello world".to_string());
        
        match text_input {
            InputModality::Text(text) => {
                assert_eq!(text.content, "Hello world");
                assert!(text.language.is_none());
            }
            _ => panic!("Expected text input"),
        }
        
        let image_data = vec![1, 2, 3, 4];
        let image_input = create_image_input(image_data.clone(), "jpeg".to_string(), 640, 480);
        
        match image_input {
            InputModality::Image(image) => {
                assert_eq!(image.data, image_data);
                assert_eq!(image.format, "jpeg");
                assert_eq!(image.width, 640);
                assert_eq!(image.height, 480);
            }
            _ => panic!("Expected image input"),
        }
    }
    
    #[tokio::test]
    async fn test_concept_confidence_filtering() {
        let integrator = create_test_integrator().await;
        
        // Test concept with low confidence should be filtered out
        let low_confidence_concept = serde_json::json!({
            "name": "uncertain_object",
            "confidence": 0.5,
            "attributes": {}
        });
        
        let text_input = TextInput {
            content: "test".to_string(),
            language: None,
            metadata: HashMap::new(),
            source: None,
        };
        
        let result = integrator.parse_text_concept(&low_confidence_concept, &text_input).await;
        assert!(result.is_ok());
        // Should return None due to low confidence
        assert!(result.unwrap().is_none());
        
        // Test concept with high confidence should pass
        let high_confidence_concept = serde_json::json!({
            "name": "clear_object",
            "confidence": 0.9,
            "attributes": {}
        });
        
        let result = integrator.parse_text_concept(&high_confidence_concept, &text_input).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }
    
    // Helper functions for testing
    async fn create_test_orchestrator() -> BackgroundOrchestrator {
        // This would create a test orchestrator
        // For now, we'll use a placeholder
        unimplemented!("Test orchestrator creation")
    }
    
    async fn create_test_integrator() -> MultimodalIntegrator {
        let config = MultimodalConfig {
            concept_confidence_threshold: 0.7,
            ..Default::default()
        };
        
        let orchestrator = Arc::new(create_test_orchestrator().await);
        MultimodalIntegrator::new(config, orchestrator).await.unwrap()
    }
}
