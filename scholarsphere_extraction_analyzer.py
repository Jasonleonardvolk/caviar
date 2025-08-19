#!/usr/bin/env python3
"""
🔬 SCHOLARSPHERE EXTRACTION DIAGNOSTICS - Surgical precision analysis
Find exactly where the PDF extraction pipeline hangs
"""

import sys
import os
import time
import traceback
import psutil
import threading
import queue
from pathlib import Path
from datetime import datetime

class ScholarSphereExtractionAnalyzer:
    """Surgical analysis of ScholarSphere extraction hanging"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.ingest_dir = self.script_dir / "ingest_pdf"
        self.pipeline_file = self.ingest_dir / "pipeline.py"
        self.timeline = []
        self.memory_snapshots = []
        
    def run_extraction_analysis(self):
        """Complete extraction analysis"""
        print("🔬 SCHOLARSPHERE EXTRACTION DIAGNOSTICS")
        print("=" * 60)
        print("🎯 Surgical analysis of PDF extraction pipeline hanging")
        print("📊 Real-time monitoring with memory and CPU tracking")
        print("=" * 60)
        
        try:
            # Phase 1: Pre-extraction environment check
            self.analyze_extraction_environment()
            
            # Phase 2: Import chain analysis
            self.analyze_extraction_imports()
            
            # Phase 3: Test with dummy PDF
            self.test_extraction_with_monitoring()
            
            # Phase 4: Analyze hanging points
            self.analyze_hanging_points()
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            traceback.print_exc()
        
        print("\\n🎯 EXTRACTION ANALYSIS COMPLETE")
    
    def log_timeline(self, event, details=""):
        """Log timeline event"""
        entry = {
            'timestamp': time.time(),
            'event': event,
            'details': details
        }
        self.timeline.append(entry)
        
        timestamp_str = datetime.fromtimestamp(entry['timestamp']).strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp_str}] {event} {details}")
    
    def analyze_extraction_environment(self):
        """Analyze extraction environment"""
        print("\\n🔍 PHASE 1: EXTRACTION ENVIRONMENT ANALYSIS")
        print("-" * 50)
        
        # Check system resources
        memory = psutil.virtual_memory()
        print(f"💾 Available memory: {memory.available // (1024**3)} GB ({memory.percent}% used)")
        
        # Check ingest_pdf directory structure
        print("\\n📁 Ingest PDF directory structure:")
        if self.ingest_dir.exists():
            critical_files = [
                "pipeline.py",
                "extractConceptsFromDocument.py", 
                "extract_blocks.py",
                "features.py",
                "spectral.py",
                "clustering.py"
            ]
            
            for file in critical_files:
                file_path = self.ingest_dir / file
                if file_path.exists():
                    size = file_path.stat().st_size
                    print(f"  ✅ {file}: {size:,} bytes")
                else:
                    print(f"  ❌ {file}: Missing")
        
        # Check data files that might cause hanging
        data_dir = self.ingest_dir / "data"
        if data_dir.exists():
            print("\\n📊 Data files analysis:")
            for file in data_dir.glob("*.json"):
                size = file.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  {file.name}: {size_mb:.1f} MB")
                if size_mb > 50:
                    print(f"    ⚠️ Large file may cause loading delays!")
    
    def analyze_extraction_imports(self):
        """Test extraction imports with timing"""
        print("\\n🔍 PHASE 2: EXTRACTION IMPORTS ANALYSIS")
        print("-" * 50)
        
        # Add ingest_pdf to path
        if str(self.ingest_dir) not in sys.path:
            sys.path.insert(0, str(self.ingest_dir))
        
        import_chain = [
            ("pipeline", "Main pipeline module"),
            ("extractConceptsFromDocument", "Universal extraction"),
            ("extract_blocks", "PDF block extraction"),
            ("features", "Feature extraction"),
            ("spectral", "Spectral analysis"),
            ("clustering", "Clustering algorithms"),
            ("scoring", "Concept scoring"),
            ("keywords", "Keyword extraction")
        ]
        
        total_import_time = 0
        
        for module_name, description in import_chain:
            try:
                self.log_timeline(f"IMPORTING_{module_name}")
                start_time = time.time()
                
                imported_module = __import__(module_name)
                
                import_time = time.time() - start_time
                total_import_time += import_time
                
                print(f"✅ {description}: {module_name} ({import_time:.3f}s)")
                
                # Check for heavy initialization
                if import_time > 5:
                    print(f"  ⚠️ Slow import - may indicate heavy initialization")
                
            except Exception as e:
                print(f"❌ {description}: {module_name} - {e}")
                self.log_timeline(f"IMPORT_FAILED_{module_name}", str(e))
        
        print(f"\\n📊 Total import time: {total_import_time:.2f}s")
        
        # Test pipeline function import specifically
        try:
            from pipeline import ingest_pdf_clean
            print("✅ ingest_pdf_clean function imported successfully")
        except Exception as e:
            print(f"❌ ingest_pdf_clean import failed: {e}")
    
    def test_extraction_with_monitoring(self):
        """Test extraction with comprehensive monitoring"""
        print("\\n🔍 PHASE 3: MONITORED EXTRACTION TEST")
        print("-" * 50)
        
        # Create a small test PDF file
        test_pdf = self.create_test_pdf()
        if not test_pdf:
            print("❌ Could not create test PDF")
            return
        
        self.log_timeline("EXTRACTION_TEST_START")
        
        # Start monitoring thread
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        monitor_thread = threading.Thread(
            target=self.monitor_system_during_extraction,
            args=(monitoring_active,),
            daemon=True
        )
        monitor_thread.start()
        
        try:
            from pipeline import ingest_pdf_clean
            
            self.log_timeline("CALLING_INGEST_PDF_CLEAN")
            print("🚀 Starting PDF extraction with monitoring...")
            
            start_time = time.time()
            
            # Call extraction with timeout monitoring
            extraction_thread = threading.Thread(
                target=self.run_extraction_with_timeout,
                args=(ingest_pdf_clean, test_pdf),
                daemon=True
            )
            extraction_thread.start()
            
            # Monitor for hanging with detailed progress
            last_memory_log = start_time
            timeout_seconds = 120  # 2 minute timeout
            
            while extraction_thread.is_alive():
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed > timeout_seconds:
                    self.log_timeline("EXTRACTION_TIMEOUT", f"{timeout_seconds}s")
                    print(f"❌ Extraction timed out after {timeout_seconds} seconds")
                    break
                
                # Log memory every 5 seconds
                if current_time - last_memory_log >= 5:
                    memory = psutil.virtual_memory()
                    self.log_timeline("MEMORY_CHECK", f"{memory.percent:.1f}% used")
                    last_memory_log = current_time
                
                # Show progress every 10 seconds
                if int(elapsed) % 10 == 0 and elapsed > 0:
                    print(f"⏳ Extraction running... {elapsed:.0f}s elapsed")
                
                time.sleep(1)
            
            extraction_thread.join(timeout=1)
            
            if extraction_thread.is_alive():
                print("⚠️ Extraction thread still running after timeout")
            
        except Exception as e:
            self.log_timeline("EXTRACTION_ERROR", str(e))
            print(f"❌ Extraction test failed: {e}")
            traceback.print_exc()
        
        finally:
            monitoring_active.clear()
            monitor_thread.join(timeout=1)
            
            # Cleanup test file
            if test_pdf and test_pdf.exists():
                test_pdf.unlink()
    
    def run_extraction_with_timeout(self, extraction_func, test_pdf):
        """Run extraction in separate thread"""
        try:
            self.log_timeline("EXTRACTION_FUNCTION_START")
            result = extraction_func(str(test_pdf))
            self.log_timeline("EXTRACTION_FUNCTION_COMPLETE", f"concepts: {result.get('concept_count', 0)}")
            print(f"✅ Extraction completed: {result.get('concept_count', 0)} concepts")
        except Exception as e:
            self.log_timeline("EXTRACTION_FUNCTION_ERROR", str(e))
            print(f"❌ Extraction function failed: {e}")
            traceback.print_exc()
    
    def monitor_system_during_extraction(self, monitoring_active):
        """Monitor system resources during extraction"""
        while monitoring_active.is_set():
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                
                snapshot = {
                    'timestamp': time.time(),
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'cpu_percent': cpu
                }
                
                self.memory_snapshots.append(snapshot)
                
                # Alert on high resource usage
                if memory.percent > 90:
                    self.log_timeline("HIGH_MEMORY_USAGE", f"{memory.percent:.1f}%")
                
                if cpu > 80:
                    self.log_timeline("HIGH_CPU_USAGE", f"{cpu:.1f}%")
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(2)  # Monitor every 2 seconds
    
    def create_test_pdf(self):
        """Create a small test PDF file"""
        try:
            # Try to find an existing PDF first
            for pdf_file in self.script_dir.glob("*.pdf"):
                if pdf_file.stat().st_size < 5 * 1024 * 1024:  # Less than 5MB
                    print(f"📄 Using existing test PDF: {pdf_file.name}")
                    return pdf_file
            
            # Create a minimal test PDF if none found
            test_content = """
            %PDF-1.4
            1 0 obj
            <<
            /Type /Catalog
            /Pages 2 0 R
            >>
            endobj
            
            2 0 obj
            <<
            /Type /Pages
            /Kids [3 0 R]
            /Count 1
            >>
            endobj
            
            3 0 obj
            <<
            /Type /Page
            /Parent 2 0 R
            /Contents 4 0 R
            >>
            endobj
            
            4 0 obj
            <<
            /Length 44
            >>
            stream
            BT
            /F1 12 Tf
            100 700 Td
            (Test document for extraction) Tj
            ET
            endstream
            endobj
            
            xref
            0 5
            0000000000 65535 f 
            0000000009 00000 n 
            0000000058 00000 n 
            0000000115 00000 n 
            0000000189 00000 n 
            trailer
            <<
            /Size 5
            /Root 1 0 R
            >>
            startxref
            285
            %%EOF
            """
            
            test_pdf = self.script_dir / "test_extraction.pdf"
            with open(test_pdf, 'w') as f:
                f.write(test_content.strip())
            
            print(f"📄 Created test PDF: {test_pdf.name}")
            return test_pdf
            
        except Exception as e:
            print(f"❌ Failed to create test PDF: {e}")
            return None
    
    def analyze_hanging_points(self):
        """Analyze where hanging occurs"""
        print("\\n🔍 PHASE 4: HANGING POINT ANALYSIS")
        print("-" * 50)
        
        print("⏱️ Timeline analysis:")
        for i, event in enumerate(self.timeline):
            timestamp = datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S.%f')[:-3]
            print(f"  {timestamp}: {event['event']} {event['details']}")
        
        # Look for gaps in timeline that indicate hanging
        print("\\n🔍 Gap analysis (potential hanging points):")
        for i in range(1, len(self.timeline)):
            time_gap = self.timeline[i]['timestamp'] - self.timeline[i-1]['timestamp']
            if time_gap > 5:  # More than 5 seconds gap
                print(f"  ⚠️ {time_gap:.1f}s gap between {self.timeline[i-1]['event']} and {self.timeline[i]['event']}")
        
        # Memory usage analysis
        if self.memory_snapshots:
            max_memory = max(s['memory_percent'] for s in self.memory_snapshots)
            min_memory = min(s['memory_percent'] for s in self.memory_snapshots)
            print(f"\\n📊 Memory usage during test: {min_memory:.1f}% - {max_memory:.1f}%")
            
            if max_memory > 80:
                print("  ⚠️ High memory usage detected - possible memory leak")
        
        # Recommendations
        print("\\n💡 HANGING POINT ANALYSIS:")
        
        timeline_events = [event['event'] for event in self.timeline]
        
        if 'EXTRACTION_TIMEOUT' in timeline_events:
            print("❌ Extraction timed out - indicates hanging in pipeline")
            
            if any('IMPORT' in event for event in timeline_events):
                print("1. Imports completed - hanging is in extraction logic")
            
            if 'EXTRACTION_FUNCTION_START' in timeline_events and 'EXTRACTION_FUNCTION_COMPLETE' not in timeline_events:
                print("2. Hanging occurs inside ingest_pdf_clean function")
                print("3. Likely culprits: PDF parsing, concept extraction, or file_storage operations")
            
        if any('HIGH_MEMORY' in event for event in timeline_events):
            print("4. Memory issues detected - check for memory leaks in extraction")
        
        print("\\n🎯 NEXT STEPS:")
        print("1. Add debugging to pipeline.py to identify exact hanging point")
        print("2. Check PDF parsing libraries for blocking operations")
        print("3. Investigate concept extraction models for hanging")
        print("4. Add timeouts to all extraction operations")

def main():
    """Run ScholarSphere extraction analysis"""
    try:
        analyzer = ScholarSphereExtractionAnalyzer()
        analyzer.run_extraction_analysis()
    except KeyboardInterrupt:
        print("\\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\\n❌ Analysis failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
