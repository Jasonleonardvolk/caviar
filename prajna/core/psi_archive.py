"""
Ψ-Archive: Arcadia Memory Architecture for Prajna
=================================================
Comprehensive cognitive logging and long-term memory system
with spectral, semantic, and metacognitive learning.

Combines:
- File-based append-only architecture (ΨArcadia)
- Advanced querying & semantic anchor indexing
- Spectral memory recall (frequency signature)
- SSE/event broadcasting (optional, async)
- Continuous pattern/learning discovery

Author: Merge of legacy & extended modules
"""

import asyncio
import logging
import time
import json
import gzip
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import deque
import hashlib

logger = logging.getLogger("prajna.psi_archive")

# ==============================
# Types & Data Structures
# ==============================
class ArchiveType(Enum):
    REFLECTION            = "reflection"
    GOAL_FORMULATION      = "goal_formulation"
    PLAN_CREATION         = "plan_creation"
    CONCEPT_SYNTHESIS     = "concept_synthesis"
    WORLD_SIMULATION      = "world_simulation"
    GHOST_DEBATE          = "ghost_debate"
    REASONING_PATH        = "reasoning_path"
    METACOGNITIVE_SESSION = "metacognitive_session"
    CONSCIOUSNESS_EVENT   = "consciousness_event"
    LEARNING_UPDATE       = "learning_update"

class CompressionLevel(Enum):
    NONE   = 0
    LOW    = 3
    MEDIUM = 6
    HIGH   = 9

@dataclass
class ArchiveRecord:
    record_id: str
    archive_type: ArchiveType
    timestamp: datetime
    data: Dict[str, Any]
    session_id: str = ""
    user_query: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    parent_records: List[str] = field(default_factory=list)
    child_records: List[str] = field(default_factory=list)
    related_records: List[str] = field(default_factory=list)
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    retention_priority: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def update_access(self):
        self.access_count += 1
        self.last_accessed = datetime.now()

@dataclass
class ArchiveQuery:
    archive_types: List[ArchiveType] = field(default_factory=list)
    date_range: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    session_id: Optional[str] = None
    confidence_range: Tuple[float, float] = (0.0, 1.0)
    contains_text: Optional[str] = None
    min_processing_time: Optional[float] = None
    max_processing_time: Optional[float] = None
    limit: int = 100
    sort_by: str = "timestamp"
    sort_order: str = "desc"

@dataclass
class ArchiveStats:
    total_records: int
    records_by_type: Dict[str, int]
    oldest_record: Optional[datetime]
    newest_record: Optional[datetime]
    total_size_bytes: int
    compression_ratio: float
    average_confidence: float
    most_accessed_types: List[Tuple[str, int]]
    learning_insights: List[str]

@dataclass
class LearningPattern:
    pattern_id: str
    pattern_type: str
    description: str
    evidence: List[str]
    confidence: float
    discovered_at: datetime
    applications: int = 0

# ==============================
# SSE Event Broadcasting
# ==============================
_event_subscribers: List[asyncio.Queue] = []

def subscribe_event_queue(queue: asyncio.Queue):
    _event_subscribers.append(queue)
    logger.info(f"Subscribed SSE client (total={len(_event_subscribers)})")

def unsubscribe_event_queue(queue: asyncio.Queue):
    try:
        _event_subscribers.remove(queue)
    except ValueError:
        pass

async def broadcast_event_to_subscribers(event: Dict[str, Any]):
    for q in list(_event_subscribers):
        try:
            await q.put(event)
        except Exception as e:
            logger.warning(f"Failed to send event: {e}")

# ==============================
# PsiArchive Main Class
# ==============================
class PsiArchive:
    """
    File-based transparency and memory system for Prajna.
    Append-only concept logs, semantic linking, fractal indexing, spectral recall,
    event broadcasting, and continuous learning.
    """

    def __init__(self, archive_path: str = "prajna_psi_archive", enable_compression: bool = True,
                 enable_learning: bool = True, max_memory_cache: int = 10000):
        self.archive_path = Path(archive_path)
        self.enable_compression = enable_compression
        self.enable_learning = enable_learning
        self.max_memory_cache = max_memory_cache

        self._initialize_archive_structure()
        self.index_path = self.archive_path / "archive_index.jsonl"
        self.records_index: List[Dict[str, Any]] = []
        self.index_by_id: Dict[str, Dict[str, Any]] = {}
        self.anchor_index: Dict[str, Set[str]] = {}
        self.memory_cache: Dict[str, ArchiveRecord] = {}
        self.cache_order = deque(maxlen=max_memory_cache)
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.learning_queue: deque = deque()
        self.archive_stats = {
            "total_archived": 0,
            "successful_retrievals": 0,
            "cache_hits": 0,
            "learning_patterns_discovered": 0,
            "total_archive_time": 0.0
        }
        self._load_or_build_index()
        self.archive_stats["total_archived"] = len(self.records_index)

        # Start learning task if enabled
        self.learning_task = None
        if self.enable_learning:
            self.learning_task = asyncio.create_task(self._continuous_learning_loop())
        logger.info(f"ΨArchive initialized at {self.archive_path}")

    def _initialize_archive_structure(self):
        self.archive_path.mkdir(parents=True, exist_ok=True)
        for t in ArchiveType:
            (self.archive_path / t.value).mkdir(parents=True, exist_ok=True)
        for aux in ["learning", "patterns", "insights"]:
            (self.archive_path / aux).mkdir(exist_ok=True)

    def _load_or_build_index(self):
        """Load archive index or build from scratch if missing/corrupt"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    for ln in f:
                        entry = json.loads(ln)
                        if 'timestamp' in entry:
                            try:
                                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                            except Exception:
                                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'].replace('Z', ''))
                        self.records_index.append(entry)
                        self.index_by_id[entry['record_id']] = entry
                        for a in entry.get('anchors', []):
                            self.anchor_index.setdefault(a, set()).add(entry['record_id'])
                return
            except Exception as e:
                logger.warning(f"Index log load failed, rebuilding... {e}")
                self.records_index.clear()
                self.index_by_id.clear()
        # No index or failed load: scan disk
        logger.info("Building archive index from disk...")
        for t in ArchiveType:
            dir_t = self.archive_path / t.value
            for fp in dir_t.rglob('*.json*'):
                try:
                    if fp.suffix == '.gz':
                        with gzip.open(fp, 'rt', encoding='utf-8') as f:
                            data = json.load(f)
                    else:
                        data = json.loads(fp.read_text(encoding='utf-8'))
                    rid = data['record_id']
                    ts = datetime.fromisoformat(data['timestamp'])
                    entry = {
                        'record_id': rid,
                        'archive_type': data['archive_type'],
                        'timestamp': ts,
                        'session_id': data.get('session_id', ''),
                        'user_query': data.get('user_query', ''),
                        'confidence': data.get('confidence', 0.0),
                        'processing_time': data.get('processing_time', 0.0),
                        'file_path': str(fp.relative_to(self.archive_path)),
                        'anchors': sorted(self._extract_anchors(data))
                    }
                    if 'frequency_signature' in data:
                        entry['frequency_signature'] = data['frequency_signature']
                    self.records_index.append(entry)
                    self.index_by_id[rid] = entry
                    for a in entry['anchors']:
                        self.anchor_index.setdefault(a, set()).add(rid)
                except Exception as e:
                    logger.error(f"Failed indexing {fp}: {e}")
        self.records_index.sort(key=lambda x: x['timestamp'])
        with open(self.index_path, 'w', encoding='utf-8') as idx:
            for e in self.records_index:
                tmp = e.copy()
                tmp['timestamp'] = tmp['timestamp'].isoformat()
                idx.write(json.dumps(tmp) + "\n")

    # ==============================
    # Spectral & Anchor Utilities
    # ==============================
    @staticmethod
    def compute_frequency_signature(phase_trace: List[float], n_bins: Optional[int] = None) -> List[float]:
        arr = np.array(phase_trace, dtype=float)
        spec = np.fft.fft(arr)
        mags = np.abs(spec)
        if n_bins and n_bins < len(mags):
            mags = mags[:n_bins]
        return mags.tolist()

    def _extract_anchors(self, record_data: Dict[str, Any]) -> Set[str]:
        anchors = set()
        try:
            text_keys = {"user_query", "description", "summary", "original_query", "prompt", "hypothesis", "final_answer", "title", "consensus"}
            for k, v in record_data.items():
                if v is None: continue
                if k in text_keys and isinstance(v, str):
                    anchors.update(self._tokenize_anchors(v))
                elif k == "goal" and isinstance(v, dict):
                    if "description" in v and isinstance(v['description'], str):
                        anchors.update(self._tokenize_anchors(v['description']))
                elif k == "outcome" and isinstance(v, dict):
                    if "consensus" in v and isinstance(v['consensus'], str):
                        anchors.update(self._tokenize_anchors(v['consensus']))
                elif k == "issues" and isinstance(v, list):
                    for issue in v:
                        if isinstance(issue, dict):
                            if "type" in issue and isinstance(issue['type'], str):
                                anchors.update(self._tokenize_anchors(issue['type']))
                            if "description" in issue and isinstance(issue['description'], str):
                                anchors.update(self._tokenize_anchors(issue['description']))
                        elif isinstance(issue, str):
                            anchors.update(self._tokenize_anchors(issue))
            if isinstance(record_data.get('user_query'), str):
                anchors.update(self._tokenize_anchors(record_data['user_query']))
        except Exception as e:
            logger.debug(f"Anchor extraction failed: {e}")
        return set(list(anchors)[:10])

    def _tokenize_anchors(self, text: str) -> Set[str]:
        tokens = []
        for token in text.replace("\n", " ").split():
            token = token.strip(".,!?;:\"()[]{}<>")
            if not token: continue
            lower = token.lower()
            if len(lower) < 3: continue
            if lower in {"the", "and", "for", "with", "this", "that", "have", "about", "from", "when", "where", "what", "which", "while", "will", "shall", "might", "could", "would", "into", "during", "been", "also", "some"}:
                continue
            tokens.append(lower)
        return set(tokens)

    # ==============================
    # Archive Logging Methods
    # ==============================
    # Each log_* method wraps _archive_record with smart param extraction
    async def log_reflection(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.REFLECTION, data,
            session_id=data.get("session_id", ""),
            user_query=data.get("original_query", ""),
            confidence=data.get("reflection_confidence", 0.0),
            processing_time=data.get("processing_time", 0.0))

    async def log_goal_formulation(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.GOAL_FORMULATION, data,
            session_id=data.get("session_id", ""),
            user_query=data.get("original_query", ""),
            confidence=data.get("goal", {}).get("confidence", 0.0),
            processing_time=data.get("processing_time", 0.0))

    async def log_plan_creation(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.PLAN_CREATION, data,
            session_id=data.get("session_id", ""),
            confidence=data.get("plan", {}).get("confidence", 0.0),
            processing_time=data.get("processing_time", 0.0))

    async def log_concept_synthesis(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.CONCEPT_SYNTHESIS, data,
            session_id=data.get("session_id", ""),
            confidence=data.get("overall_coherence", 0.0),
            processing_time=data.get("synthesis_time", 0.0))

    async def log_world_simulation(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.WORLD_SIMULATION, data,
            session_id=data.get("session_id", ""),
            confidence=data.get("consistency_score", 0.0),
            processing_time=data.get("simulation_time", 0.0))

    async def log_ghost_debate(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.GHOST_DEBATE, data,
            session_id=data.get("session_id", ""),
            user_query=data.get("prompt", ""),
            confidence=data.get("outcome", {}).get("confidence_score", 0.0),
            processing_time=data.get("metrics", {}).get("debate_time", 0.0))

    async def log_reasoning_path(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.REASONING_PATH, data,
            session_id=data.get("session_id", ""),
            confidence=data.get("confidence", 0.0),
            processing_time=data.get("reasoning_time", 0.0))

    async def log_metacognitive_session(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.METACOGNITIVE_SESSION, data,
            session_id=data.get("session_id", ""),
            user_query=data.get("original_query", ""),
            confidence=data.get("final_confidence", 0.0),
            processing_time=data.get("total_processing_time", 0.0))

    async def log_consciousness_event(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.CONSCIOUSNESS_EVENT, data,
            session_id=data.get("session_id", ""),
            confidence=data.get("consciousness_level", 0.0),
            processing_time=data.get("processing_time", 0.0))

    async def log_learning_update(self, data: Dict[str, Any]) -> str:
        return await self._archive_record(ArchiveType.LEARNING_UPDATE, data,
            session_id=data.get("session_id", ""),
            confidence=data.get("confidence", 0.0))

    async def _archive_record(self, archive_type: ArchiveType, data: Dict[str, Any],
                               session_id: str = "", user_query: str = "",
                               confidence: float = 0.0, processing_time: float = 0.0) -> str:
        """Core archiving function: appends a new record to the memory log"""
        start_time = time.time()
        try:
            record_id = self._generate_record_id(archive_type, data)
            now = datetime.now()
            # ArchiveRecord object (for cache)
            record = ArchiveRecord(
                record_id=record_id,
                archive_type=archive_type,
                timestamp=now,
                data=data,
                session_id=session_id,
                user_query=user_query,
                confidence=confidence,
                processing_time=processing_time
            )
            date_str = now.strftime("%Y/%m/%d")
            record_dir = self.archive_path / archive_type.value / date_str
            record_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{record_id}.json"
            if self.enable_compression:
                filename += ".gz"
            full_path = record_dir / filename

            payload = {
                "record_id": record.record_id,
                "archive_type": record.archive_type.value,
                "timestamp": record.timestamp.isoformat(),
                "data": record.data,
                "session_id": record.session_id,
                "user_query": record.user_query,
                "confidence": record.confidence,
                "processing_time": record.processing_time
            }
            # Optionally spectral memory
            if isinstance(data.get('phase_trace'), list):
                payload['frequency_signature'] = self.compute_frequency_signature(data['phase_trace'])

            # Save record file
            if self.enable_compression:
                with gzip.open(full_path, 'wt', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)
            else:
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)

            data_hash = hashlib.sha256(json.dumps(record.data, sort_keys=True).encode()).hexdigest()[:16]
            anchors = sorted(list(self._extract_anchors({**record.data, "user_query": user_query})))
            entry = {
                'record_id': record_id,
                'archive_type': archive_type.value,
                'timestamp': now,
                'session_id': session_id,
                'user_query': user_query,
                'confidence': confidence,
                'processing_time': processing_time,
                'file_path': str(full_path.relative_to(self.archive_path)),
                'anchors': anchors,
                'data_hash': data_hash
            }
            if 'frequency_signature' in payload:
                entry['frequency_signature'] = payload['frequency_signature']

            self.records_index.append(entry)
            self.index_by_id[record_id] = entry
            for a in anchors:
                self.anchor_index.setdefault(a, set()).add(record_id)
            try:
                with open(self.index_path, 'a', encoding='utf-8') as idx:
                    tmp = entry.copy()
                    if isinstance(tmp['timestamp'], datetime):
                        tmp['timestamp'] = tmp['timestamp'].isoformat()
                    json.dump(tmp, idx)
                    idx.write("\n")
            except Exception as e:
                logger.warning(f"Failed to append to index log: {e}")

            self._add_to_cache(record)
            self.archive_stats["total_archived"] += 1
            self.archive_stats["total_archive_time"] += (time.time() - start_time)

            # Broadcast event (if SSE used)
            await broadcast_event_to_subscribers({
                'event_type': archive_type.value,
                'record_id': record_id,
                'data': data
            })

            # Learning queue
            if self.enable_learning:
                self.learning_queue.append(record_id)
            # Back-propagate session if session record
            if archive_type == ArchiveType.METACOGNITIVE_SESSION:
                try:
                    related_ids = data.get("archive_records", [])
                    for rid in related_ids:
                        if rid in self.index_by_id:
                            self.index_by_id[rid]['session_id'] = record.session_id
                            if rid in self.memory_cache:
                                self.memory_cache[rid].session_id = record.session_id
                except Exception as e:
                    logger.warning(f"Failed to back-propagate session ID: {e}")
            return record_id
        except Exception as e:
            logger.error(f"Archiving failed for {archive_type.value}: {e}")
            return ""

    def _generate_record_id(self, archive_type: ArchiveType, data: Dict[str, Any]) -> str:
        ts = datetime.now().isoformat()
        data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        ts_safe = ts.replace(':', '-')
        return f"{archive_type.value}_{ts_safe}_{data_hash}"

    # ==============================
    # Querying & Retrieval
    # ==============================
    async def query_archive(self, query: ArchiveQuery) -> List[ArchiveRecord]:
        results_meta = []
        for e in self.records_index:
            if query.archive_types and e['archive_type'] not in [t.value for t in query.archive_types]:
                continue
            if query.date_range[0] and e['timestamp'] < query.date_range[0]:
                continue
            if query.date_range[1] and e['timestamp'] > query.date_range[1]:
                continue
            if query.session_id and e['session_id'] != query.session_id:
                continue
            if not (query.confidence_range[0] <= e['confidence'] <= query.confidence_range[1]):
                continue
            if query.contains_text:
                text = query.contains_text.lower()
                if text not in e.get('user_query','').lower() and not any(text in a for a in e.get('anchors',[])):
                    continue
            if query.min_processing_time and e['processing_time'] < query.min_processing_time:
                continue
            if query.max_processing_time and e['processing_time'] > query.max_processing_time:
                continue
            results_meta.append(e)
        reverse = query.sort_order.lower() == 'desc'
        if query.sort_by in ['timestamp','confidence','processing_time','access_count']:
            results_meta.sort(key=lambda x: x.get(query.sort_by,0) or 0, reverse=reverse)
        else:
            results_meta.sort(key=lambda x: x['timestamp'], reverse=reverse)
        results_meta = results_meta[:query.limit]
        records = []
        for e in results_meta:
            rec = await self._load_record(e['record_id'], e['file_path'])
            if rec:
                records.append(rec)
        self.archive_stats['successful_retrievals'] += len(records)
        return records

    async def recall_by_spectrum(self, query_signature: List[float], top_k: int = 5) -> List[ArchiveRecord]:
        sims = []
        q = np.array(query_signature, dtype=float)
        qnorm = np.linalg.norm(q)
        for e in self.records_index:
            fs = e.get('frequency_signature')
            if not fs:
                continue
            f = np.array(fs, dtype=float)
            ln = min(len(f), len(q))
            if ln < 1:
                continue
            score = float(np.dot(q[:ln], f[:ln]) / (np.linalg.norm(f[:ln]) * qnorm + 1e-9))
            sims.append((score, e['record_id'], e['file_path']))
        sims.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, rid, fp in sims[:top_k]:
            rec = await self._load_record(rid, fp)
            if rec:
                results.append(rec)
        return results

    async def _load_record(self, record_id: str, file_path: str) -> Optional[ArchiveRecord]:
        if record_id in self.memory_cache:
            rec = self.memory_cache[record_id]
            rec.update_access()
            self.archive_stats['cache_hits'] += 1
            return rec
        try:
            full = self.archive_path / file_path
            if str(full).endswith('.gz'):
                with gzip.open(full, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = json.loads(full.read_text(encoding='utf-8'))
            rec = ArchiveRecord(
                record_id=data['record_id'],
                archive_type=ArchiveType(data['archive_type']),
                timestamp=datetime.fromisoformat(data['timestamp']),
                data=data.get('data',{}),
                session_id=data.get('session_id',''),
                user_query=data.get('user_query',''),
                confidence=data.get('confidence',0.0),
                processing_time=data.get('processing_time',0.0)
            )
            self._add_to_cache(rec)
            return rec
        except Exception as e:
            logger.error(f"Failed to load record {record_id}: {e}")
            return None

    def _add_to_cache(self, record: ArchiveRecord):
        if len(self.memory_cache) >= self.max_memory_cache:
            old = self.cache_order.popleft()
            self.memory_cache.pop(old, None)
        self.memory_cache[record.record_id] = record
        self.cache_order.append(record.record_id)

    # ==============================
    # Learning, Pattern Analysis
    # ==============================
    async def get_metacognitive_history(self, filters: Dict[str, Any] = None) -> List[ArchiveRecord]:
        query = ArchiveQuery()
        if filters:
            at = filters.get('archive_types')
            if at:
                query.archive_types = [ArchiveType(x) for x in at]
            tf = filters.get('timeframe')
            if tf:
                now = datetime.now()
                if tf == 'last_hour':
                    query.date_range = (now - timedelta(hours=1), None)
                elif tf == 'last_day':
                    query.date_range = (now - timedelta(days=1), None)
                elif tf == 'last_week':
                    query.date_range = (now - timedelta(weeks=1), None)
            sid = filters.get('session_id')
            if sid:
                query.session_id = sid
            mc = filters.get('min_confidence')
            if mc is not None:
                query.confidence_range = (mc, 1.0)
            lim = filters.get('limit')
            if lim:
                query.limit = lim
        return await self.query_archive(query)

    async def analyze_performance_patterns(self) -> List[LearningPattern]:
        patterns = []
        try:
            recent = await self.query_archive(ArchiveQuery(
                date_range=(datetime.now() - timedelta(days=7), None),
                limit=10000
            ))
            # Placeholder for pattern analysis (extend with your logic!)
            return patterns
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return patterns

    async def _continuous_learning_loop(self):
        while True:
            try:
                if not self.learning_queue:
                    await asyncio.sleep(60)
                    continue
                if len(self.learning_queue) > 100:
                    await self.analyze_performance_patterns()
                    self.learning_queue.clear()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuous learning error: {e}")
                await asyncio.sleep(300)

    # ==============================
    # Maintenance & Stats
    # ==============================
    async def cleanup_old_records(self, days_to_keep: int = 365):
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        to_del = [e for e in self.records_index if e['timestamp'] < cutoff]
        for e in to_del:
            fp = self.archive_path / e['file_path']
            try:
                fp.unlink()
            except:
                pass
            self.records_index.remove(e)
            self.index_by_id.pop(e['record_id'], None)
        with open(self.index_path, 'w', encoding='utf-8') as idx:
            for e in self.records_index:
                tmp = e.copy()
                tmp['timestamp'] = tmp['timestamp'].isoformat()
                idx.write(json.dumps(tmp) + "\n")
        logger.info(f"Cleaned {len(to_del)} old records")

    async def get_archive_stats(self) -> ArchiveStats:
        total = len(self.records_index)
        types = {}
        for e in self.records_index:
            t = e['archive_type']
            types[t] = types.get(t, 0) + 1
        times = [e['timestamp'] for e in self.records_index]
        oldest = min(times) if times else None
        newest = max(times) if times else None
        avg_conf = (sum(e.get('confidence', 0.0) for e in self.records_index) / total) if total > 0 else 0.0
        access_by_type = {}
        for rec in self.memory_cache.values():
            tval = rec.archive_type.value
            access_by_type[tval] = access_by_type.get(tval, 0) + rec.access_count
        most_accessed = sorted(access_by_type.items(), key=lambda kv: kv[1], reverse=True)[:5]
        total_size = sum(f.stat().st_size for f in self.archive_path.rglob("*") if f.is_file())
        compression_ratio = 0.7 if self.enable_compression else 1.0
        learning_insights = [
            f"Discovered {len(self.learned_patterns)} learning patterns",
            f"Cache hit rate: {self.archive_stats['cache_hits'] / max(1, self.archive_stats['total_archived']):.1%}",
            f"Average archival time: {self.archive_stats['total_archive_time'] / max(1, self.archive_stats['total_archived']):.3f}s"
        ]
        return ArchiveStats(
            total_records=total,
            records_by_type=types,
            oldest_record=oldest,
            newest_record=newest,
            total_size_bytes=total_size,
            compression_ratio=compression_ratio,
            average_confidence=avg_conf,
            most_accessed_types=most_accessed,
            learning_insights=learning_insights
        )

    # ==============================
    # Self-test / Entrypoint
    # ==============================
if __name__ == "__main__":
    # Minimal async self-test
    async def test_psi_archive():
        archive = PsiArchive("test_archive", enable_learning=True, enable_compression=False)
        test_query = "What is the capital of France?"
        goal_data = {"session_id": "session_test_1", "original_query": test_query,
                     "goal": {"description": "Find capital of France", "confidence": 0.9},
                     "processing_time": 0.1}
        plan_data = {"session_id": "session_test_1", "plan": {"confidence": 0.8}, "processing_time": 0.2}
        debate_data = {"session_id": "session_test_1", "prompt": test_query,
                       "outcome": {"confidence_score": 0.95}, "metrics": {"debate_time": 0.3}}
        reflection_data = {"session_id": "session_test_1", "original_query": test_query,
                           "reflection_confidence": 0.85, "processing_time": 0.1,
                           "issues": [{"type": "accuracy", "description": "Check factual correctness"}]}
        session_data = {"session_id": "session_test_1", "original_query": test_query,
                        "final_answer": "Paris is the capital of France.", "final_confidence": 0.95,
                        "total_processing_time": 1.0, "archive_records": []}
        # Log each event
        gid = await archive.log_goal_formulation(goal_data)
        session_data["archive_records"].append(gid)
        pid = await archive.log_plan_creation(plan_data)
        session_data["archive_records"].append(pid)
        did = await archive.log_ghost_debate(debate_data)
        session_data["archive_records"].append(did)
        rid = await archive.log_reflection(reflection_data)
        session_data["archive_records"].append(rid)
        sid = await archive.log_metacognitive_session(session_data)
        print(f"Logged records: {gid}, {pid}, {did}, {rid}, {sid}")
        # Query recent records
        recent = await archive.get_metacognitive_history({"timeframe": "last_day"})
        print(f"Retrieved {len(recent)} records from archive (last day).")
        stats = await archive.get_archive_stats()
        print(f"Total records: {stats.total_records}, Types: {stats.records_by_type}")
        await archive.cleanup_old_records(days_to_keep=0)  # clean up for test
    asyncio.run(test_psi_archive())
