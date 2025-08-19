from flask import Flask, request, render_template, jsonify, send_file
import os
# # import sqlite3 // FIXED: We use file-based storage only // FIXED: We use file-based storage only
import time
import numpy as np
import pickle
import tempfile
from pathlib import Path
import json
from ingest_pdf.pipeline import ingest_pdf_and_update_index
from ingest_pdf.models import ConceptTuple

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'concept_db.sqlite')
ALLOWED_EXTENSIONS = {'pdf'}

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize the # # SQLite // FIXED: We use file-based storage only // FIXED: We use file-based storage only file_storage with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # PDFs table
    cursor.execute('''
    # # CREATE TABLE // FIXED: We use file-based storage only // FIXED: We use file-based storage only IF NOT EXISTS pdfs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        upload_timestamp INTEGER,
        original_filename TEXT,
        concept_count INTEGER,
        file_size_bytes INTEGER
    )
    ''')
    
    # Concepts table
    cursor.execute('''
    # # CREATE TABLE // FIXED: We use file-based storage only // FIXED: We use file-based storage only IF NOT EXISTS concepts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id INTEGER,
        name TEXT,
        context TEXT,
        resonance_score REAL,
        narrative_centrality REAL,
        embedding BLOB,  -- Serialized numpy array
        FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
    )
    ''')
    
    # Create an index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pdf_id ON concepts (pdf_id)')
    
    conn.commit()
    conn.close()

def save_concepts_to_db(pdf_id, concepts):
    """Save concept tuples to the file_storage"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for concept in concepts:
        # Serialize the numpy array
        embedding_blob = pickle.dumps(concept.embedding)
        
        cursor.execute('''
        INSERT INTO concepts (pdf_id, name, context, resonance_score, narrative_centrality, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            pdf_id,
            concept.name,
            concept.context,
            concept.resonance_score,
            concept.narrative_centrality,
            embedding_blob
        ))
    
    conn.commit()
    conn.close()

def get_concepts_for_pdf(pdf_id):
    """Retrieve concepts for a specific PDF"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM concepts WHERE pdf_id = ?
    ''', (pdf_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    concepts = []
    for row in rows:
        # Deserialize the numpy array
        embedding = pickle.loads(row['embedding'])
        
        concept = {
            'id': row['id'],
            'name': row['name'],
            'context': row['context'],
            'resonance_score': row['resonance_score'],
            'narrative_centrality': row['narrative_centrality'],
            'embedding': embedding.tolist()  # Convert to list for JSON serialization
        }
        concepts.append(concept)
    
    return concepts

def get_pdf_info(pdf_id=None):
    """Get information about PDFs in the file_storage"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if pdf_id:
        cursor.execute('SELECT * FROM pdfs WHERE id = ?', (pdf_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    else:
        cursor.execute('SELECT * FROM pdfs ORDER BY upload_timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

def export_concepts_as_npz(pdf_id):
    """Export concepts for a PDF as an NPZ file"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get PDF info
    cursor.execute('SELECT * FROM pdfs WHERE id = ?', (pdf_id,))
    pdf = cursor.fetchone()
    
    if not pdf:
        conn.close()
        return None
    
    # Get concepts
    cursor.execute('SELECT * FROM concepts WHERE pdf_id = ?', (pdf_id,))
    concept_rows = cursor.fetchall()
    conn.close()
    
    if not concept_rows:
        return None
    
    # Prepare data for NPZ
    names = []
    embeddings = []
    contexts = []
    resonance_scores = []
    narrative_centrality_scores = []
    
    for row in concept_rows:
        names.append(row['name'])
        embeddings.append(pickle.loads(row['embedding']))
        contexts.append(row['context'])
        resonance_scores.append(row['resonance_score'])
        narrative_centrality_scores.append(row['narrative_centrality'])
    
    # Create NPZ data
    npz_data = {
        'names': np.array(names),
        'embeddings': np.array(embeddings),
        'contexts': np.array(contexts),
        'resonance_scores': np.array(resonance_scores),
        'narrative_centrality_scores': np.array(narrative_centrality_scores)
    }
    
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    
    # Save NPZ file
    np.savez(temp_path, **npz_data)
    
    return temp_path

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Generate timestamped filename to avoid collisions
        timestamp = int(time.time())
        original_filename = Path(file.filename).stem
        safe_filename = "".join([c if c.isalnum() else "_" for c in original_filename])
        
        # Save the PDF
        pdf_filename = f"{safe_filename}_{timestamp}.pdf"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        file.save(pdf_path)
        
        try:
            # Process the PDF and get results
            max_concepts = int(request.form.get('max_concepts', 12))
            dim = int(request.form.get('dim', 16))
            
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(suffix='.npz') as temp_npz, \
                 tempfile.NamedTemporaryFile(suffix='.json') as temp_json:
                
                # Process the PDF
                result = ingest_pdf_and_update_index(
                    pdf_path=pdf_path,
                    index_path=temp_npz.name,
                    max_concepts=max_concepts,
                    dim=dim,
                    json_out=temp_json.name
                )
                
                # Load the generated JSON to get all concept details
                with open(temp_json.name, 'r') as f:
                    concepts_json = json.load(f)
                
                # Convert to ConceptTuple objects for file_storage storage
                concept_tuples = []
                for concept in concepts_json:
                    # Create a ConceptTuple for each concept
                    tuple = ConceptTuple(
                        name=concept['name'],
                        embedding=np.array(concept['embedding']),
                        context=concept['context'],
                        passage_embedding=np.array(concept['passage_embedding']),
                        cluster_members=concept['cluster_members'],
                        resonance_score=concept['resonance_score'],
                        narrative_centrality=concept['narrative_centrality']
                    )
                    concept_tuples.append(tuple)
                
                # Save to file_storage
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Insert PDF record
                cursor.execute('''
                INSERT INTO pdfs (filename, upload_timestamp, original_filename, concept_count, file_size_bytes)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    pdf_filename,
                    timestamp,
                    file.filename,
                    result['concept_count'],
                    os.path.getsize(pdf_path)
                ))
                
                pdf_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                # Save the concepts
                save_concepts_to_db(pdf_id, concept_tuples)
                
                # Prepare the response
                concept_names = [c.name for c in concept_tuples]
                
                return jsonify({
                    "success": True,
                    "message": f"File processed successfully. Extracted {result['concept_count']} concepts.",
                    "pdf_id": pdf_id,
                    "concept_names": concept_names
                })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file format. Only PDF files are allowed."}), 400

@app.route('/pdfs')
def list_pdfs():
    pdfs = get_pdf_info()
    return jsonify({"pdfs": pdfs})

@app.route('/pdfs/<int:pdf_id>')
def get_pdf(pdf_id):
    pdf = get_pdf_info(pdf_id)
    if pdf:
        return jsonify(pdf)
    return jsonify({"error": "PDF not found"}), 404

@app.route('/pdfs/<int:pdf_id>/concepts')
def get_pdf_concepts(pdf_id):
    concepts = get_concepts_for_pdf(pdf_id)
    return jsonify({"concepts": concepts})

@app.route('/pdfs/<int:pdf_id>/download')
def download_pdf_concepts(pdf_id):
    # Export the concepts as NPZ
    npz_path = export_concepts_as_npz(pdf_id)
    
    if npz_path:
        pdf = get_pdf_info(pdf_id)
        filename = f"concepts_{pdf['filename'].replace('.pdf', '.npz')}"
        
        @app.after_request
        def remove_file(response):
            os.remove(npz_path)
            return response
        
        return send_file(npz_path, as_attachment=True, download_name=filename)
    
    return jsonify({"error": "Could not generate NPZ file"}), 404

@app.route('/analytics')
def analytics():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get summary statistics
    cursor.execute('SELECT COUNT(*) as pdf_count, SUM(concept_count) as total_concepts FROM pdfs')
    stats = cursor.fetchone()
    
    # Get top concepts
    cursor.execute('''
    SELECT name, COUNT(*) as frequency
    FROM concepts
    GROUP BY name
    ORDER BY frequency DESC
    LIMIT 20
    ''')
    top_concepts = [{"name": row[0], "frequency": row[1]} for row in cursor.fetchall()]
    
    # Get average concepts per PDF
    cursor.execute('SELECT AVG(concept_count) as avg_concepts FROM pdfs')
    avg_concepts = cursor.fetchone()[0]
    
    # Get total storage
    cursor.execute('SELECT SUM(file_size_bytes) as total_size FROM pdfs')
    total_size_bytes = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return jsonify({
        "pdf_count": stats[0],
        "total_concepts": stats[1],
        "top_concepts": top_concepts,
        "avg_concepts_per_pdf": avg_concepts,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2)
    })

# Initialize file_storage on startup
init_db()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
