| variant | Spearman rho | Kendall tau | top-3 | same top-3 set | max rank move |
|---|---|---|---|---|---|
| default | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| pair score additive (v1 form) | 0.901 | 0.744 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 3 (gemini-3-flash-preview) |
| coverage v1 form | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| position tolerance 2.5% diag | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| position tolerance 10% diag | 0.945 | 0.846 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 3 (gemini-3-flash-preview) |
| orientation saturates 15 deg | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| orientation saturates 90 deg | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| constraint tolerances x0.5 | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| constraint tolerances x2 | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| constraint tolerances x5 | 0.995 | 0.974 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 1 (gemini-3-flash-preview) |
| weights 0.2/0.3/0.5 (v1) | 0.995 | 0.974 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 1 (kimi-k2.5) |
| weights 0.5/0.25/0.25 | 0.951 | 0.872 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 3 (gemini-3-flash-preview) |
| weights 0.25/0.5/0.25 | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
| weights 0.25/0.25/0.5 | 0.978 | 0.923 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 2 (gpt-4.1) |
| Geometry only | 0.989 | 0.949 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 1 (claude-opus-4-6) |
| Semantic only | 0.973 | 0.897 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 2 (claude-opus-4-6) |
| all v1-like choices | 0.912 | 0.769 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 3 (gpt-4.1) |
| Sem = mean over constraints | 0.989 | 0.949 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 1 (gemini-3-flash-preview) |
| Sem = mean over constraint kinds | 0.995 | 0.974 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 1 (claude-opus-4-6) |
| Sem = mean over prompt sentences | 1.000 | 1.000 | gemini-3.1-pro-preview, gpt-5.4, gemini-2.5-pro | True | 0 (gemini-3.1-pro-preview) |
