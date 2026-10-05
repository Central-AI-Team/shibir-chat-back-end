#!/usr/bin/env python3
"""
debug_pipeline.py — Pipeline Step-by-Step Debugger (fixed)
===========================================================
Steps:
  1  normalize()           chunker.py
  2  detect_language()     query_rewriter.py
  3  _is_conversational()  qa_service.py
  4  classify_intent()     intent_classifier.py
  5  expand_query()        query_rewriter.py  (LLM)
  6  embed_texts()         embedder.py
  7  Chroma + Rerank       retriever.py  ← FIX: stage_b দেখায়
  8  relevance gate        qa_service.py
  9  generate_answer()     generator.py  (LLM)

Run:
  cd shibir-chat-back-end-dev
  python scripts/debug_pipeline.py
"""
from __future__ import annotations
import os, sys, time, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(ROOT, ".env"))

R="\033[0m"; B="\033[1m"; DIM="\033[2m"
STEP="\033[38;5;39m"; OK="\033[38;5;82m"; WARN="\033[38;5;214m"
ERR="\033[38;5;196m"; IN="\033[38;5;159m"; VAL="\033[38;5;255m"
W=64

def _box(t,c=None):
    c=c or STEP; bar="─"*W; pad=W-len(t)-2
    print(f"\n{c}{B}┌{bar}┐{R}\n{c}{B}│ {t}{' '*max(0,pad)}│{R}\n{c}{B}└{bar}┘{R}")
def _label(k,v,c="\033[38;5;229m"): print(f"  {c}{B}{k:<18}{R} {VAL}{v}{R}")
def _sub(t):
    [print(f"  {DIM}{l}{R}") for l in textwrap.wrap(t,W-4)]
def _ok(m):   print(f"  {OK}✓ {m}{R}")
def _warn(m): print(f"  {WARN}⚠ {m}{R}")
def _err(m):  print(f"  {ERR}✗ {m}{R}")
def _sep():   print(f"  {DIM}{'·'*W}{R}")
def _ms(ms):  return f"{ms:.0f}ms" if ms<1000 else f"{ms/1000:.2f}s"
def _imp(n):
    import importlib
    try: return importlib.import_module(n)
    except Exception as e: _err(f"import {n}: {e}"); return None

# ── Step 1 ────────────────────────────────────────────────────────────────────
def step_normalize(raw):
    _box("STEP 1 — normalize()  [chunker.py]")
    _label("INPUT",repr(raw),IN)
    mod=_imp("app.rag.chunker")
    if mod is None: return raw
    t0=time.perf_counter(); out=mod.normalize(raw); ms=(time.perf_counter()-t0)*1000
    _label("OUTPUT",repr(out))
    _warn("changed") if raw!=out else _ok("No change")
    _label("TIME",_ms(ms),DIM); return out

# ── Step 2 ────────────────────────────────────────────────────────────────────
def step_detect(query):
    _box("STEP 2 — detect_language()  [query_rewriter.py]")
    _label("INPUT",repr(query),IN)
    mod=_imp("app.rag.query_rewriter")
    if mod is None: return "UNKNOWN"
    t0=time.perf_counter(); route=mod.detect_language(query); ms=(time.perf_counter()-t0)*1000
    desc={"BENGALI":"Bengali script → direct + synonym","BANGLISH":"Banglish → transliterate","ENGLISH_ARABIC":"EN/AR → translate to Bengali"}
    _label("ROUTE",route); _sub(desc.get(route,"")); _label("TIME",_ms(ms),DIM)
    return route

# ── Step 3 ────────────────────────────────────────────────────────────────────
def step_conv(query):
    _box("STEP 3 — _is_conversational()  [qa_service.py]")
    _label("INPUT",repr(query),IN)
    mod=_imp("app.services.qa_service")
    if mod is None: return False
    t0=time.perf_counter(); r=mod._is_conversational(query); ms=(time.perf_counter()-t0)*1000
    _label("IS CONVERSATIONAL",str(r))
    if r: _warn("Greeting detected → PIPELINE SHORT-CIRCUITS")
    else: _ok("Not conversational — continues")
    _label("TIME",_ms(ms),DIM); return r

# ── Step 4 ────────────────────────────────────────────────────────────────────
def step_intent(query):
    _box("STEP 4 — classify_intent()  [intent_classifier.py]")
    _label("INPUT",repr(query),IN)
    mod=_imp("app.services.intent_classifier")
    if mod is None: return "QA"
    t0=time.perf_counter()
    try: intent=mod.classify_intent(query,False)  # has_active_roleplay_session
    except Exception as e: _err(str(e)); return "QA"
    ms=(time.perf_counter()-t0)*1000
    desc={"QA":"Normal Q&A → retrieval","NOTE":"Note → note service","SUGGESTION":"Suggestion service","ROLEPLAY":"Roleplay service"}
    _label("INTENT",intent); _sub(desc.get(intent,""))
    if intent!="QA": _warn("Routes AWAY from RAG")
    else: _ok("QA → continues")
    _label("TIME",_ms(ms),DIM); return intent

# ── Step 5 ────────────────────────────────────────────────────────────────────
def step_expand(query):
    _box("STEP 5 — expand_query()  [query_rewriter.py]  🔑 LLM")
    _label("INPUT",repr(query),IN)
    mod=_imp("app.rag.query_rewriter")
    if mod is None: return (query,)
    mod.expand_query.cache_clear()
    t0=time.perf_counter()
    try:
        v=mod.expand_query(query); ms=(time.perf_counter()-t0)*1000
        _label(f"VARIANTS ({len(v)})","")
        for i,s in enumerate(v):
            is_orig=(i==len(v)-1 and i>0 and not any("\u0980"<=c<="\u09ff" for c in s))
            tag=" ← canonical (bn)" if i==0 else (" ← original" if is_orig else "")
            print(f"    [{i}] {VAL}{s}{R}{DIM}{tag}{R}")
        if mod.get_fallback_count()>0: _warn("LLM failed — raw fallback")
        else: _ok("LLM rewrite OK")
        _label("TIME",_ms(ms),DIM); return v
    except Exception as e:
        _err(str(e)); _label("TIME",_ms((time.perf_counter()-t0)*1000),DIM); return (query,)

# ── Step 6 ────────────────────────────────────────────────────────────────────
def step_embed(variants):
    _box("STEP 6 — embed_texts()  [embedder.py]  (bge-m3)")
    _label("INPUT",f"{len(variants)} string(s)",IN)
    mod=_imp("app.rag.embedder")
    if mod is None: return []
    t0=time.perf_counter()
    try:
        embs=mod.embed_texts(list(variants)); ms=(time.perf_counter()-t0)*1000
        for i,e in enumerate(embs): _label(f"  [{i}] dim",str(len(e)))
        _ok(f"Embedded {len(embs)} variant(s)"); _label("TIME",_ms(ms),DIM); return embs
    except Exception as e:
        _err(str(e)); _label("TIME",_ms((time.perf_counter()-t0)*1000),DIM); return []

# ── Step 7 — FIX: stage_b (reranked final) দেখাচ্ছে ─────────────────────────
def step_retrieve(query):
    _box("STEP 7 — Chroma Search + Rerank  [retriever.py]")
    _label("INPUT",repr(query),IN)
    mod_r=_imp("app.rag.retriever"); mod_c=_imp("app.core.config")
    if mod_r is None or mod_c is None: return [],[]
    cfg=mod_c.settings
    t0=time.perf_counter()
    try:
        # retrieve_stages → (stage_a=candidates, stage_b=reranked_final)
        stage_a, stage_b = mod_r.retrieve_stages(query)
        ms=(time.perf_counter()-t0)*1000

        # Chroma raw output (stage_a) — sim score only
        _label("fetch_k",str(cfg.fetch_k)); _label("min_sim",str(cfg.min_similarity))
        _label("CANDIDATES",f"{len(stage_a)} chunks from Chroma")
        _sep()
        print(f"  {DIM}   SIM    BOOK{R}")
        for i,c in enumerate(stage_a[:5]):
            print(f"  [{i:02d}] {OK}{c.similarity:.3f}{R}  {DIM}{c.book[:45]}{R}")
        if len(stage_a)>5: _sub(f"… {len(stage_a)-5} more")

        # Reranked final (stage_b) — rerank score ← এটাই সঠিক
        _sep()
        _label("AFTER RERANK",f"{len(stage_b)} chunks kept  (top_k={cfg.top_k})")
        _label("min_rerank",str(cfg.min_rerank_score))
        _sep()
        print(f"  {DIM}  RERANK    SIM    STATUS    BOOK{R}")
        for i,c in enumerate(stage_b):
            score=c.rerank_score or 0.0; passed=score>=cfg.min_rerank_score
            sc=OK if passed else WARN; tag="✓ PASS" if passed else "✗ FAIL"
            print(f"  [{i}] {sc}{score:.3f}{R}    {DIM}{c.similarity:.3f}{R}    {sc}{tag}{R}    {VAL}{c.book[:28]}{R}")
            _sub(f"     └─ {(c.content or '')[:80].replace(chr(10),' ')}…")

        _label("TIME",_ms(ms),DIM)
        return stage_a, stage_b
    except Exception as e:
        _err(str(e)); import traceback; traceback.print_exc()
        _label("TIME",_ms((time.perf_counter()-t0)*1000),DIM); return [],[]

# ── Step 8 ────────────────────────────────────────────────────────────────────
def step_gate(final):
    _box("STEP 8 — relevance gate  [qa_service.py]")
    mod_c=_imp("app.core.config"); thr=mod_c.settings.min_rerank_score if mod_c else 0.5
    top=final[0].rerank_score if final else None
    _label("threshold",str(thr),IN)
    _label("top rerank",f"{top:.3f}" if top else "None")
    grounded=bool(final) and top is not None and top>=thr
    if grounded:
        n=sum(1 for c in final if c.rerank_score and c.rerank_score>=thr)
        _ok(f"GROUNDED — {n} chunk(s) pass"); _ok("→ generate_answer() with context")
    else:
        _warn(f"NOT GROUNDED (top={top}, thr={thr})")
        _warn("→ generate_answer() with EMPTY context")
        _warn("→ 'বইতে তথ্য পাওয়া যায়নি'")
    return grounded

# ── Step 9 ────────────────────────────────────────────────────────────────────
def step_generate(query, final, grounded):
    _box("STEP 9 — generate_answer()  [generator.py]  🔑 LLM")
    mod=_imp("app.rag.generator"); ms_=_imp("app.schemas.query"); mc=_imp("app.core.config")
    if mod is None or ms_ is None: return ""
    thr=mc.settings.min_rerank_score if mc else 0.5
    cits=[]
    if grounded:
        for c in final:
            if c.rerank_score and c.rerank_score>=thr:
                cits.append(ms_.Citation(book=c.book,chapter=c.chapter,source_db=c.source_db,
                    content=c.content,similarity=round(c.similarity,4),
                    rerank_score=round(c.rerank_score,4)))
    _label("CITATIONS",f"{len(cits)} passed to LLM",IN)
    for i,c in enumerate(cits): print(f"    [{i}] rerank={c.rerank_score}  {c.book[:40]}")
    t0=time.perf_counter()
    try:
        ans=mod.generate_answer(query,cits); ms=(time.perf_counter()-t0)*1000
        _label("TIME",_ms(ms),DIM); _sep()
        print(f"\n  {B}{OK}FINAL ANSWER:{R}")
        for line in ans.split("\n"): print(f"  {VAL}{line}{R}")
        return ans
    except Exception as e:
        _err(str(e)); _label("TIME",_ms((time.perf_counter()-t0)*1000),DIM); return ""

# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(raw):
    t=time.perf_counter()
    print(f"\n{B}{'═'*(W+2)}{R}\n{B}  PIPELINE DEBUG — {raw}{R}\n{B}{'═'*(W+2)}{R}")
    q=step_normalize(raw); route=step_detect(q)
    if step_conv(q):
        _box("PIPELINE ENDED — conversational",WARN); _warn("No retrieval, no LLM."); return
    intent=step_intent(q)
    if intent!="QA":
        _box(f"PIPELINE ENDED — intent={intent}",WARN); _warn("Non-RAG service."); return
    v=step_expand(q); step_embed(v)
    ca,fi=step_retrieve(q); grounded=step_gate(fi); step_generate(q,fi,grounded)
    _box("PIPELINE COMPLETE",OK)
    _label("Total time",_ms((time.perf_counter()-t)*1000))
    _label("Route",route); _label("Candidates",str(len(ca)))
    _label("After rerank",str(len(fi))); _label("Grounded",str(grounded)); print()

def main():
    print(f"\n{B}╔{'═'*W}╗\n║{'  Pipeline Debugger (fixed ✓)':^{W}}║\n║{'  Bengali · Banglish · English · Arabic':^{W}}║\n╚{'═'*W}╝{R}\n\n  {DIM}type query → see every step | 'q' → quit{R}\n")
    while True:
        try: q=input(f"{B}Query >{R} ").strip()
        except (EOFError,KeyboardInterrupt): print(); break
        if not q: continue
        if q.lower() in ("q","exit","quit"): break
        run_pipeline(q)

if __name__=="__main__":
    main()
