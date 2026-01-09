# ScoreStat  
### A Statistical Framework for Event-Level Sports Performance Analysis

## Overview
ScoreStat is a sports analytics platform designed to model matches as sequences of discrete events and derive meaningful statistical insights from event-level data. The project focuses on correctness, statistical reasoning, and explainable analytics rather than UI-heavy features.

The initial implementation targets cricket, where each ball is treated as an atomic event. From this granular representation, player and team performance metrics are computed systematically.

##  Motivation
In local tournaments, scoring is often done manually and only final results are preserved. This leads to loss of detailed performance data and limits statistical analysis. ScoreStat addresses this problem by storing complete event-level data, enabling deeper quantitative insights and historical performance tracking.

##  Core Concept
- A match is modeled as a **sequence of discrete events**
- Each **ball** represents one atomic event
- All statistics are **derived**, not hardcoded
- The database serves as the **single source of truth**

## Features (Cricket – Phase 1)
- Ball-by-ball scoring engine
- Correct handling of overs, strike rotation, and wickets
- Derived match summaries
- Player performance metrics:
  - Runs, strike rate, averages
  - Consistency using variance
- Team performance evaluation using weighted aggregation
- Explainable match outcome tendency estimation


---

## 🗂️ System Architecture
