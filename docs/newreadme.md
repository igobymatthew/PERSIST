# PERSIST vXR — Composite Systems Architecture (Publication Layout)

```mermaid
flowchart TD
%% =========================================================
%% PERSIST vXR Composite Architecture (Publication Layout)
%% =========================================================

%% --- Core Physiology ---
subgraph CORE["Core Engine — Physiological Subsystems"]
    HK["Viability Kernel / EnsembleShield"]
    HS["Homeostat / Constraint Manager"]
    EM["Empowerment / Surprise / RND"]
    WM["Latent World Model / Dreamer RSSM"]
    EWC["EWC / Rehearsal Buffer"]
end

%% --- Reflexive Meta Layer ---
subgraph META["Reflexive & Regulatory Layer"]
    ML["Meta-Learner (Hormonal Adaptation)"]
    FIRE["Fire Cycle (Regeneration)"]
    GA["Genetic Engine (Evolutionary Search)"]
    OV["Overseer-LoRA (Executive Control)"]
end

%% --- Affective Layer ---
subgraph AFFECT["Affective & Developmental Systems"]
    EEA["Emotional Equilibrium Agent++"]
    LSM["LifeStage Manager"]
end

%% --- Ecosystem Fabric ---
subgraph ECO["Ecosystem Fabric — Biodiversity Fabric Simulator (BFS)"]
    ENV["Environment / Energy Fields"]
    SPEC1["Species A — Forager"]
    SPEC2["Species B — Predator"]
    SPEC3["Species C — Mutualist"]
    SPEC4["Species D — Scavenger"]
    POPS["Population Monitor"]
    FIREG["Global Fire Cycle"]
    GAG["GA / NSGA-II Evolution"]
end

%% --- Telemetry & Infrastructure ---
subgraph OPS["Telemetry & Infrastructure"]
    TELE["Prometheus Telemetry"]
    BUD["Budget Meter / Maintenance Manager"]
    ADV["Adversarial Trainer"]
    ANALYTICS["Analytics / Visualization"]
end

%% --- Core loops ---
HK --> HS --> EM --> WM --> EWC --> ML
ML --> FIRE --> GA --> OV
OV --> HK
OV --> EEA --> LSM
LSM --> SPEC1 & SPEC2 & SPEC3 & SPEC4
SPEC1 & SPEC2 & SPEC3 & SPEC4 --> POPS --> OV
OV --> FIREG --> ENV
FIREG --> SPEC1 & SPEC2 & SPEC3 & SPEC4
GAG --> SPEC1 & SPEC2 & SPEC3 & SPEC4

%% --- Feedback & telemetry ---
TELE --> OV
BUD --> HS
ADV --> FIRE
POPS --> TELE --> ANALYTICS --> OV

%% --- Clean style for publication ---
classDef core fill:#0b0b0b,stroke:#666,stroke-width:0.8px,color:#f6f6f6;
classDef meta fill:#141414,stroke:#777,stroke-width:0.8px,color:#ffffff;
classDef affect fill:#1b0b1b,stroke:#9370db,stroke-width:0.8px,color:#eee5ff;
classDef eco fill:#081808,stroke:#6b8e23,stroke-width:0.8px,color:#eaffea;
classDef ops fill:#151515,stroke:#888,stroke-width:0.8px,color:#fff8e1;

class CORE core
class META meta
class AFFECT affect
class ECO eco
class OPS ops