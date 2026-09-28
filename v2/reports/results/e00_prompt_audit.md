# E00. Prompt scaffolding audit

Script: `v2/experiments/e00_prompt_audit.py` (regexes: the released `scripts/audit_prompts.py`, unchanged).

## Regex flags (high-risk = any of the four)

| prompt set | prompts | high-risk | shape count | coordinate vector | formula/assignment | trig/Cartesian |
|---|---|---|---|---|---|---|
| v1 released | 75 | 60 | 48 | 27 | 9 | 18 |
| v2 task text | 75 | 0 | 0 | 0 | 0 | 0 |
| v2 arm `json` (task-specific text) | 75 | 0 | 0 | 0 | 0 | 0 |
| v2 arm `neutral` (task-specific text) | 75 | 0 | 0 | 0 | 0 | 0 |
| v2 arm `tool` (task-specific text) | 75 | 0 | 0 | 0 | 0 | 0 |
| v2 arm `mates` (task-specific text) | 75 | 0 | 0 | 0 | 0 | 0 |
| v2 arm `schema` (task-specific text) | 75 | 0 | 0 | 0 | 0 | 0 |
| v2 arm `cadquery` (task-specific text) | 75 | 0 | 0 | 0 | 0 | 0 |
| v2 arm `probe` (task-specific text) | 75 | 0 | 0 | 0 | 0 | 0 |

Shared instruction blocks (identical for every task; audited once):

- output contract: coordinate_vector
- tool instruction: no flag
- cadquery instruction: coordinate_vector
- mates instruction + language reference + examples: coordinate_vector

These matches are not task values: the regex's character class admits the output contract's literal `{"shapes": [...]}` and the CadQuery instruction's dict index `parts[0]`; the mates reference and its two worked examples (assemblies outside the benchmark) contain coordinate vectors by design.

## Verbatim coordinate overlap (% of nonzero reference coordinate components found as numbers in the prompt)

| prompts | by occurrence | by distinct value per case |
|---|---|---|
| v1 released | 22.7 | 10.5 |
| v2 | 13.2 | 7.2 |

Highest v2 families (hand review: are these base inputs that coincide with coordinates, or derived values?):

- Gantry Crane Assembly: 71.4% (v1: 71.4%)
- Pipe Manifold: 51.2% (v1: 48.8%)
- Suspension Bridge: 50.0% (v1: 80.9%)
- Honeycomb Lattice: 40.8% (v1: 57.3%)
- Phyllotaxis Disc: 33.3% (v1: 33.3%)
- Clock Tower Mechanism: 23.2% (v1: 37.6%)
- Cannonball Pyramid: 20.6% (v1: 27.2%)
- Flanged Pipe Joint: 18.8% (v1: 23.4%)
