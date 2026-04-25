# Viraltest v2 — YouTube Script (<2 minutes)

## Storyboard

### Shot 1: Hook (0:00–0:10)
**Visual:** Split screen — left: scrolling Instagram feed, right: an LLM terminal making decisions
**Voiceover:** "What if an AI agent could learn to run your Instagram account — not from a prompt, but by discovering the rules of the world itself?"
**On-screen text:** "Viraltest v2 — World Modeling for Instagram"

### Shot 2: The Problem (0:10–0:25)
**Visual:** Stats flying in — "$250B creator economy" (Goldman Sachs 2025), "73% burnout" (Awin 2024), "67M creators" 
**Voiceover:** "67 million creators compete for attention. 73% burn out. The algorithm changes constantly. No one tells you the rules."
**Citation badge:** Goldman Sachs 2025 · Awin 2024

### Shot 3: The Environment (0:25–0:50)
**Visual:** Animated diagram — agent receives sparse observation → calls tools → gets data → plans day
**Voiceover:** "We built a 30-day Instagram simulation. The agent sees almost nothing — just energy, followers, and last reward. To learn, it must use 8 discoverable tools: query trends, check competitors, test plans before committing."
**On-screen text:** "8 tools · 5 audience segments · 7 competitor archetypes · 30-day horizon"
**Citation badge:** Buffer 9.6M · Sprout Social 2B · Van Dongen 2003

### Shot 4: The Science (0:50–1:10)
**Visual:** Side-by-side comparison tables showing env constants vs. source data
**Voiceover:** "Every number comes from real research. Engagement rates from Socialinsider's 31-million post study. Peak hours from Buffer's 9.6-million post analysis. Sleep decay from a 2003 Sleep journal paper. Algorithm signals from Instagram's own head, Adam Mosseri."
**Citation badge:** Mosseri Jan-2025 · Socialinsider 2026 · PMID 12683469

### Shot 5: Training Results (1:10–1:30)
**Visual:** `plots/before_after.png` (grouped bars), `plots/reward_curve.png` (per-step curves)
**Voiceover:** "Even a tiny 4-billion-parameter Gemma running locally beats a heuristic baseline on all three tasks once it gets to use the tool catalog. The hardest task — `monthly_competitive` — collapses the baseline because anti-gaming gates punish single-content-type strategies."
**On-screen text:** "Gemma E4B · local llama.cpp · zero training tokens spent"

### Shot 6: Theme Fit + Close (1:30–1:50)
**Visual:** Theme #3.1 checklist being checked off — tool discovery, partial observability, persistent state, causal reasoning, multi-step workflow
**Voiceover:** "This is Theme 3.1: World Modeling. Real tool interaction. Persistent state across months. Causal reasoning through counterfactual feedback. Not a toy — a simulation grounded in science."
**On-screen text:** "All sources: RESEARCH.md · Code: github.com/... · Try it: HF Spaces"

---

**Total runtime:** ~1:50
**Music:** Upbeat lo-fi instrumental (no lyrics)
**Aspect ratio:** 16:9 landscape
