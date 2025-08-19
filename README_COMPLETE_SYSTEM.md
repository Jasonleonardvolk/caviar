# 🌍🎬 Universal Multimodal Concept Extraction System

**Complete AI-powered concept extraction across ALL academic domains and content types**

Transform any scholarly content—documents, videos, lectures, papers—into structured knowledge with automatic concept discovery and cross-domain intelligence.

## 🎯 System Overview

This system provides **universal concept extraction** that works across:

### 📚 **Text Content (Universal Extraction)**
- **PDF Documents**: Scientific papers, textbooks, reports
- **Academic Articles**: Research papers, essays, theses  
- **Presentation Slides**: Lecture notes, conference slides

### 🎬 **Video Content (Multimodal Extraction)**
- **Educational Videos**: University lectures, online courses
- **Documentary Films**: Cultural, historical, scientific content
- **Demonstrations**: Lab work, experiments, tutorials
- **Presentations**: Recorded talks, webinars, conferences

### 🌍 **Universal Domain Coverage**
- **Sciences**: Physics, Biology, Chemistry, Computer Science
- **Humanities**: Philosophy, Literature, History, Linguistics
- **Arts**: Art History, Music Theory, Visual Arts, Film Studies
- **Mathematics**: Pure Math, Applied Math, Statistics
- **Social Sciences**: Psychology, Sociology, Economics, Anthropology

## 🧬 Key Features

### **Universal Text Extraction**
- **Triple-Method Approach**: YAKE + KeyBERT + spaCy NER
- **Universal Embeddings**: Works across all academic domains
- **Wikidata Integration**: Links concepts to universal knowledge base
- **Auto-Discovery**: Finds domain-specific terminology automatically

### **Multimodal Video Extraction**  
- **Audio Transcription**: Whisper/Vosk speech-to-text
- **Visual Analysis**: BLIP image captioning + Detectron2 object detection
- **OCR Text Extraction**: Reads text from slides and visuals
- **Temporal Alignment**: Aligns concepts across audio/visual modalities
- **Cross-Modal Fusion**: Combines and scores concepts from all sources

### **Intelligent Auto-Prefill Database**
- **Self-Growing**: Automatically discovers and adds new concepts
- **Cross-Domain**: Learns concepts from any academic field
- **Quality Filtering**: Only adds high-confidence concepts
- **Domain Classification**: Automatically categorizes concepts by field
- **Alias Generation**: Creates meaningful concept variations

### **Integration & Pipeline**
- **Unified Architecture**: Seamless text + video processing
- **ScholarSphere Integration**: Upload any content type
- **ConceptMesh Integration**: Structured concept relationships
- **Batch Processing**: Handle large content collections

## 🚀 Quick Start

### **1. Install Universal Text Extraction**
```bash
python setup_universal_extraction.py
```

### **2. Install Video Processing (Optional)**
```bash
python setup_video_extraction.py
```

### **3. Test the System**
```bash
# Test text extraction
python demo_universal_extraction.py

# Test video extraction  
python test_video_extraction.py

# Test complete system
python demo_complete_system.py
```

### **4. Process Your Content**
```python
# Text documents
from ingest_pdf.extractConceptsFromDocument import extractConceptsFromDocument
concepts = extractConceptsFromDocument(document_text)

# Video files
from ingest_pdf.extractConceptsFromVideo import extractConceptsFromVideo
concepts = extractConceptsFromVideo("lecture.mp4")

# YouTube videos
from ingest_pdf.extractConceptsFromVideo import extractConceptsFromYouTubeVideo
concepts = extractConceptsFromYouTubeVideo("https://youtube.com/watch?v=...")
```

## 🎬 Video Processing Capabilities

### **Audio Analysis**
- **Whisper ASR**: State-of-the-art speech recognition
- **Technical Terminology**: Handles academic jargon and domain-specific terms
- **Multiple Languages**: Support for international content
- **Timestamp Alignment**: Links concepts to specific video moments

### **Visual Analysis**
- **Object Detection**: Identifies equipment, diagrams, people, settings
- **Image Captioning**: Describes visual scenes and contexts
- **OCR Text Extraction**: Reads slides, equations, diagrams, captions
- **Scene Understanding**: Contextual analysis of visual content

### **Multimodal Fusion**
- **Temporal Alignment**: Matches audio and visual concepts by timestamp
- **Cross-Reference Boosting**: Higher scores for concepts found in multiple modalities
- **Context Integration**: Combines spoken content with visual evidence
- **Redundancy Elimination**: Intelligent deduplication across modalities

## 📊 Example Results

### **Physics Lecture Video**
```json
[
  {
    "name": "Quantum Entanglement",
    "score": 0.95,
    "modalities": ["audio", "visual"],
    "timestamps": ["00:10:15", "00:45:30"],
    "context": "Mentioned in speech + shown on slide"
  },
  {
    "name": "Bell's Theorem", 
    "score": 0.88,
    "modalities": ["audio"],
    "timestamps": ["00:42:10"],
    "context": "Explained during lecture"
  }
]
```

### **Art History Documentary**
```json
[
  {
    "name": "Chiaroscuro Technique",
    "score": 0.92,
    "modalities": ["audio", "visual"],
    "timestamps": ["00:05:20", "00:12:45"],
    "context": "Discussed + demonstrated in paintings"
  },
  {
    "name": "Leonardo da Vinci",
    "score": 0.89,
    "modalities": ["audio", "visual"],
    "timestamps": ["00:08:30", "00:15:10"],
    "context": "Named in narration + shown in artwork"
  }
]
```

## 🌍 Universal Coverage Examples

| Domain | Text Concepts | Video Concepts |
|--------|---------------|----------------|
| **Physics** | Quantum mechanics, Relativity theory, Particle physics | Laboratory demonstrations, Equation derivations, Experimental setups |
| **Philosophy** | Phenomenology, Existentialism, Epistemology | Philosophical discussions, Historical contexts, Thought experiments |
| **Art History** | Renaissance art, Baroque painting, Modern movements | Artwork analysis, Museum tours, Artist biographies |
| **Biology** | DNA replication, Evolution, Cell biology | Microscopy footage, Dissections, Ecological studies |
| **Mathematics** | Category theory, Topology, Analysis | Proof demonstrations, Geometric visualizations, Problem solving |

## 🧬 Auto-Prefill Database Evolution

The system continuously learns and grows:

1. **Initial Seed**: 50+ carefully curated concepts across all domains
2. **Document Processing**: Discovers domain-specific terminology from papers
3. **Video Analysis**: Learns from visual and spoken academic content  
4. **Cross-Referencing**: Validates concepts across multiple sources
5. **Quality Control**: Only adds high-confidence, meaningful concepts
6. **Domain Expansion**: Automatically grows into new academic fields

### **Database Growth Simulation**
- **Start**: 50 universal seed concepts
- **After 10 PDFs**: +50 concepts (domain-specific terms)
- **After 5 videos**: +70 concepts (multimodal discoveries)  
- **After 100 items**: +500-1000 concepts (comprehensive coverage)

## 🎯 Real-World Applications

### **University Course Analysis**
- Process textbooks, papers, lecture videos, and slides
- Generate comprehensive course concept maps
- Identify knowledge gaps and connections
- Support curriculum development

### **Research Literature Review**  
- Analyze papers, conference videos, and books
- Extract research methodology and findings
- Map concept relationships and evolution
- Support systematic reviews

### **Educational Content Creation**
- Process diverse source materials
- Extract key concepts for curriculum design
- Ensure comprehensive topic coverage
- Support adaptive learning systems

### **Cultural Studies Research**
- Analyze texts, documentaries, and artistic content
- Map cultural movements and influences
- Cross-reference historical and artistic concepts
- Support interdisciplinary research

## 🔧 System Architecture

### **Modular Design**
- **Text Pipeline**: Universal document processing
- **Video Pipeline**: Multimodal content analysis
- **Fusion Layer**: Cross-modal concept integration
- **Database Layer**: Auto-growing concept repository
- **API Layer**: Unified interface for all content types

### **Scalability Features**
- **Batch Processing**: Handle large content collections
- **Parallel Processing**: Multi-threaded analysis
- **GPU Acceleration**: Optional for faster processing
- **Cloud Integration**: Scalable deployment options

### **Quality Assurance**
- **Multiple Validation**: Cross-modal concept verification
- **Confidence Scoring**: Quality-based concept ranking
- **Domain Filtering**: Prevents low-quality concept pollution
- **Human-in-Loop**: Optional manual concept curation

## 📋 Dependencies

### **Core Requirements**
- Python 3.8+
- PyTorch (for deep learning models)
- Transformers (for BERT/BLIP models)
- spaCy (for NLP processing)

### **Text Processing**
- YAKE (keyword extraction)
- KeyBERT (semantic keyphrase extraction)
- sentence-transformers (universal embeddings)

### **Video Processing**
- OpenAI Whisper (speech recognition)
- OpenCV (video processing)
- Detectron2 (object detection)
- pytesseract (OCR)

### **System Integration**
- FFmpeg (video processing)
- Tesseract OCR (text recognition)
- Wikidata API (entity linking)

## 🎉 Getting Started

1. **Clone the repository**
2. **Run setup scripts** for your needed capabilities
3. **Test with sample content** using demo scripts
4. **Process your academic content** through ScholarSphere
5. **Watch your concept database grow** automatically

The system is designed to work out-of-the-box with minimal configuration while providing powerful customization options for advanced users.

---

**🌍 Universal. 🎬 Multimodal. 🧬 Self-Growing. 🚀 Ready for any academic content.**
