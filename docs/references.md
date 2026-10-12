# References

Papers referenced in the Kolam Labs project. PDFs, where kept, are local only (not tracked in git).

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

---

## [5] Kolam array grammars and picture languages

**Authors:** Gift Siromoney and colleagues; K.G. Subramanian et al.  
**Links:**  
- A New Model of Array Grammar (kolam array grammars, L-models): https://arxiv.org/pdf/1407.8337  
- Gift Siromoney, theoretical computer science and kolam: https://www.cmi.ac.in/gift/TheoreticalComputerScience.htm  
- Tree Adjunct Kolam Array Languages: https://journal.iisc.ac.in/index.php/iisc/article/view/3945  
- A unifying approach to picture grammars: https://arxiv.org/pdf/0910.2829

**Relevance:** Kolams as two-dimensional arrays rewritten by grammar rules. The substitution system (spec §11.2) is this idea on our gate grid: a block rule per dot and a seam rule per gap.

---

## [6] L-systems and kolams

**Authors:** Przemyslaw Prusinkiewicz; Jean-Paul Allouche, Guy Allouche, Jeffrey Shallit  
**Links:**  
- Graphical applications of L-systems (1986): https://www.algorithmicbotany.org/papers/graphical.gi86.html  
- Kolam, Sierpinski curve and monoid morphisms (Ann. Inst. Fourier, 2006): https://aif.centre-mersenne.org/articles/10.5802/aif.2235/  
- Yale fractals panorama, kolams: https://users.math.yale.edu/public_html/People/frame/Fractals/Panorama/Art/Kolams/Kolams.html  
- CPython turtledemo `lindenmayer.py` (Krishna's Anklets): https://android.googlesource.com/platform/external/python/cpython3/+/cfca4a6774/Lib/turtledemo/lindenmayer.py

**Relevance:** Turtle L-systems draw classic kolams (Krishna's Anklets) generation by generation. Spec §11.2 keeps the rewriting idea but rewrites gaps instead of turtle moves, so every output is an exact label.

---

## [7] Mirror curves

**Author:** Paulus Gerdes  
**Links:**  
- Lunda Geometry: Mirror Curves, Designs, Knots, Polyominoes, Patterns, Symmetries: https://www.sahistory.org.za/sites/default/files/archive-files3/paulus_gerdes_lunda_geometry_mirror_curves_desbook4you.pdf  
- On m-canonic mirror curves: https://www.mi.sanu.ac.rs/vismath/gerd1/index.html

**Relevance:** A curve bouncing between two-sided mirrors placed on a dot grid. Our gap digits are such mirrors (wall = mirror, cross = no mirror). The switching system (spec §11.3) places mirrors to merge curves into one, together with [4].

---

## [8] One-stroke pulli kolam dataset

**Author:** Seshadri Sivakumar  
**Link:** https://www.kaggle.com/datasets/seshadrisivakumar/one-stroke-dotted-pulli-kolam

**Relevance:** Real one-stroke kolams from the gate-switching algorithm of [4]; a possible source for WFC pieces (spec §11.4) and a test set.

---

## [9] Wave function collapse

**Authors:** Maxim Gumin (algorithm); Markovian analysis, arXiv 2509.09919  
**Links:**  
- https://github.com/mxgmn/WaveFunctionCollapse  
- https://arxiv.org/pdf/2509.09919

**Relevance:** Fills a grid with tiles learned from an example, least-constrained-last. Spec §11.4 learns dot pieces from a photographed kolam.

---

## [10] Hridaya kolam for even-ordered dot patterns

**Link:** https://pith.science/paper/2507.02874 (arXiv 2507.02874)

**Relevance:** Constructive rules for a family of one-stroke kolams on growing dot grids; a candidate for another rule-based system.

---

## [11] Pix2Seq: a language modeling framework for object detection

**Authors:** Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, Geoffrey Hinton  
**Published:** ICLR 2022  
**Link:** https://arxiv.org/abs/2109.10852

**Relevance:** Objects written as sequences of discrete coordinate tokens and read off an image by a decoder. The gap-code token layout (spec §12.4) does the same with gap addresses and digits.

---

## [12] PaliGemma

**Authors:** Lucas Beyer et al., Google DeepMind  
**Link:** https://arxiv.org/abs/2407.07726

**Relevance:** A pretrained vision-language model that adds special location tokens to its vocabulary for detection. The same route fits the 59 gap-code tokens when fine-tuning a ready-made model (spec §12.5).

