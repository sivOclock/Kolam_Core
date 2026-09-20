# References

Papers referenced in the Kolam Labs project. PDFs are stored locally in `ref_papers/` (not tracked in git).

---

## [1] Sequential Encoding of Tamil Kolam Patterns

**Authors:** Timothy M. Waring  
**Published:** *Forma*, 27, 83–92, 2012 (received 2011, accepted 2012)  
**Institution:** School of Economics, University of Maine  
**Link:** https://www.researchgate.net/publication/234116828_Sequential_Encoding_of_Tamil_Kolam_Patterns

**Relevance:** Introduces a global typology of kolam types and an expanded sequential gestural lexicon for square loop kolams (SLK). Directly related to this project's sequence encoding approach — establishes the idea of decomposing kolam patterns into ordered gesture sequences and encoding them as a language. The distinction between orthogonal (von Neumann) and diagonal (Moore neighborhood) gestures is foundational.

---

## [2] Fundamental Study on Design System of Kolam Pattern

**Authors:** Kiwamu Yanagisawa, Shojiro Nagata  
**Published:** *Forma*, 22, 31–46, 2007  
**Institutions:** Kobe Design University; InterVision Institute  
**Link:** https://www.researchgate.net/publication/237442288_Fundamental_Study_on_Design_System_of_Kolam_Pattern

**Relevance:** Formalizes kolam drawing rules (5 rules including closed-loop, space-filling, smooth curves, no right-angle bends) and introduces numeric/binary conversion of kolam patterns via crossing/uncrossing at inclined grid intersections. The exhaustive analysis of one-stroke patterns on 1-5-1 and 1-7-1 dot arrays is directly applicable to this project's isomorphism and dataset generation goals.

---

## [3] Kolam Simulation using Angles at Lattice Points

**Authors:** Tulasi Bharathi, Shailaja D. Sharma, Nithin Nagaraj  
**Published:** arXiv:2307.02144 [cs.IT], July 2023  
**DOI:** https://doi.org/10.48550/arXiv.2307.02144  
**Link:** https://arxiv.org/abs/2307.02144

**Relevance:** Proposes encoding kolam paths as sequences of 4 angular symbols at lattice points, unique up to cyclic permutations. Demonstrates single-loop kolam simulation via turtle-move sequences in Python. This angular encoding scheme is a direct alternative to the coordinate-based `[li,lj]` sequence format used in this project and informs the tokenization strategy for model training.

---

## [4] An Algorithm for One-Stroke Kolam Generation using a Gating Structure

**Authors:** Seshadri Sivakumar, Shyamala Sivakumar  
**Published:** *npj Heritage Science*, 14(1), March 2026  
**DOI:** https://doi.org/10.1038/s40494-026-02310-3  
**License:** CC BY 4.0  
**Link:** https://www.researchgate.net/publication/401728191_An_algorithm_for_one-stroke_kolam_generation_using_a_gating_structure

**Relevance:** Develops a novel iterative gate-switching algorithm to autonomously generate symmetric one-stroke kolams around arbitrary anchor-dot grids. The gating structure (cross/uncross decisions at each dot) maps directly to the binary local decision problem this project aims to learn. Relevant to Phase 4 (synthetic data generation) and the L-system inspired generator roadmap item.
