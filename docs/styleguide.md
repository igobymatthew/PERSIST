
* * *

# 🧬 PERSIST vXR -- Visual Identity & Documentation Style Guide

  

Revision 2025-10-21 -- for internal and publication use
* * *

## 1. Design Ethos

  

Aesthetic principle:

  

> "Cold precision shaped by living systems."

  

Everything should feel engineered yet organic -- a convergence of biotechnical precision and ecological depth.

Think: black-body circuitry, mycelial networks, neural tissue, polished steel, and ritual minimalism.

  

Core values:

- Modularity: every component diagrammed like a living cell.
- Continuity: circular or layered composition--feedback over hierarchy.
- Contrast: darkness with faint bioluminescent hues (signal amidst entropy).
- Legibility: research-paper minimalism, no unnecessary ornament.
* * *

## 2. Color System

| 

Layer / Domain

 | 

Hex

 | 

Role

 | 

Impression

 | 
| ---- | ---- | ---- | ----  |
| 

Core Engine

 | 

#0a0a0a

 | 

Background / code blocks

 | 

Carbon, structure, body

 | 
| 

Meta / Reflexive

 | 

#141414

 | 

Layer boundaries

 | 

Shadowed cortex

 | 
| 

Affect / Development

 | 

#4b0082 → #9370db

 | 

Emotional and temporal gradients

 | 

Violet: memory, imagination

 | 
| 

Ecosystem / BFS

 | 

#1a3d1a → #6b8e23

 | 

Life, ecology, growth

 | 

Green bioluminescence

 | 
| 

Ops / Telemetry

 | 

#333333 + #ffd700 accents

 | 

Infrastructure / nervous system

 | 

Pulse of data

 | 
| 

Highlight (Energy)

 | 

#00ff99

 | 

Active feedback, empowerment, energy surge

 | 

Vital signal

 | 
| 

Accent (Entropy / Fire)

 | 

#ff3300

 | 

Fire Cycle / regeneration marker

 | 

Destruction--renewal contrast

 | 

Usage rules:

- Backgrounds dark and matte; content luminous and restrained.
- Only 1 accent per figure (empowerment or fire).
- Use soft gradients vertically to imply flow and depth.
* * *

## 3. Typography

| 

Context

 | 

Typeface

 | 

Notes

 | 
| ---- | ---- | ----  |
| 

Headings / Titles

 | 

Space Grotesk or Inter Tight

 | 

Technical but elegant; high x-height.

 | 
| 

Body / Captions

 | 

IBM Plex Sans or Source Sans 3

 | 

Academic readability.

 | 
| 

Code / CLI / Diagrams

 | 

JetBrains Mono

 | 

Monospaced clarity; evokes engineering.

 | 
| 

Logotype / Wordmark

 | 

"PERSIST" in uppercase Space Grotesk, letter-spaced 120%, matte white on black.

 |  | 

Typography should echo scientific precision rather than branding flash.

Use consistent 1.25× line height and wide margins to mimic journal layouts.

* * *

## 4. Layout & Composition

- Grid system: 8-point modular grid (baseline rhythm for all diagrams and tables).
- Alignment: prefer vertical stacks over lateral sprawl (like anatomical charts).
- Whitespace: breathe--dark voids around subsystems increase perceived intelligence.
- Section dividers: use thin muted lines (rgba(255,255,255,0.1)) or double em-dash ⸻ for print.
- Figures: label numerically (Fig. 1, Fig. 2) with precise captions referencing module names.
* * *

## 5. Diagram Conventions

  

Shape semantics:

- Rounded rectangles: subsystems, biological analogs.
- Hexagons: decision or control nodes (e.g., Overseer triggers).
- Ellipses: continuous phenomena (Empowerment, Energy fields).
- Double borders: meta-learning or recursive processes.
- Color logic: mirror section palette above.

  

Connector semantics:

- Solid line: deterministic data flow.
- Dashed line: probabilistic / meta influence.
- Gradient line: feedback loop across timescales.
- Arrowheads: always directed toward entropy reduction or empowerment gain.

  

Label style:

- Sentence case, small sans-serif, opacity 0.8
- Never use bold inside diagrams -- rely on proximity and alignment.
* * *

## 6. Documentation Voice & Tone

  

For papers and READMEs:

- Academic tone with restrained lyricism: precise first, evocative second.
- Write in continuous prose -- avoid bullet-point overload in conceptual sections.
- Integrate equations elegantly in-line (\\(\\mathcal{E}(s_t)\\) not blocky).
- Each section should mirror biological progression: cell → organ → organism → ecology.

  

For commit messages & internal notes:

- Use minimal imperative tense:

    - Added Empowerment buffer for cross-agent viability replay.

    - FireCycle: reduced threshold variance by 0.05.

  

For tooltips or CLI:

- Short, cryptic, elegant. Examples:

    - > Fire Cycle initiated.

    - > Homeostat reaching metabolic limit.

    - > Overseer: parameter entropy rising.
* * *

## 7. Iconography & Motifs

- Main icon: concentric circles (3 layers) -- representing viability, homeostasis, empowerment.
- Secondary motif: flame-inside-leaf -- symbolizing controlled regeneration.
- Motion direction: clockwise spirals (persistence loops).
- Animation speed: slow drift, 0.2--0.4 Hz cycles for meditative motion (if using in web assets).
- Textural base: microscopic noise, like analog film grain -- evokes organic computation.
* * *

## 8. Presentation Standards

  

Poster / Slide Background: #0a0a0a matte black.

Text color: #f5f5f5 or near-white.

Highlight elements: bright desaturated green (#00ffa2) or violet depending on layer.

Image ratio: 16 : 9 for screen, 3 : 4 for print.

Title placement: top left, small caps, no shadows.

Figure numbering: sequential across docs (F1--F12 for all diagrams).

* * *

## 9. Implementation for GitHub Pages or Docs Site

  

If you plan to deploy docs via GitHub Pages / Docusaurus / MkDocs:
    
    
    theme:
      name: 'material'
      palette:
        - scheme: default
          primary: 'black'
          accent: 'lime'
          background: '#0a0a0a'
      font:
        text: 'IBM Plex Sans'
        code: 'JetBrains Mono'
    markdown_extensions:
      - admonition
      - pymdownx.superfences
      - pymdownx.highlight
      - pymdownx.inlinehilite

> Enable Mermaid extension for diagrams, and dark-mode only for brand coherence.

* * *

## 10. Example Title Slide Layout (Text Template)
    
    
    ───────────────────────────────────────────────
          PERSIST vXR -- Adaptive Ecosystem Engine
    ───────────────────────────────────────────────
       A research framework for intrinsic viability,
       empowerment, and self-governing AI systems.
    
       Matthew [igobymatthew] · 2025
    ───────────────────────────────────────────────

Font sizes scale: Title 48 pt · Subtitle 20 pt · Body 14 pt · Margins 80 px.

* * *

## 11. Signature Tagline

  

> "WIP"

  

Use once per major document; never as a banner.

* * *
