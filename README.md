# MedFlow AI

> AI-powered decision support system for optimizing medical resource allocation across healthcare facilities

## Problem Statement

Public healthcare systems face a critical challenge: **inefficient allocation of medical resources**. Hospitals struggle to match resource availability (ventilators, oxygen cylinders, medications, diagnostic kits, staff) with real-time patient needs.

This problem intensifies during outbreaks and surges, where manual decision-making and outdated data lead to:
- Underutilization in some facilities
- Critical shortages in others  
- Suboptimal patient outcomes
- Inefficient cost management

**The gap?** Lack of intelligent systems that can forecast demand, predict shortages, and recommend optimal resource distribution strategies.

## Solution

MedFlow AI is an adaptive AI agent that:

1. **Analyzes** current resource distribution across healthcare facilities
2. **Predicts** future demand and potential shortages using time-series forecasting
3. **Optimizes** resource allocation through mathematical optimization
4. **Adapts** to decision-maker preferences through reinforcement learning from user interactions
5. **Explains** recommendations with clear, actionable reasoning

### Key Capabilities

- Real-time resource status monitoring across 100+ hospitals
-  7-14 day demand forecasting for critical medical resources
- Shortage risk detection and early warning system
- Multi-objective optimization (cost, coverage, fairness, urgency)
- Preference learning that adapts to user priorities over time
- Explainable recommendations with similar case retrieval

### Architecture 
<img width="5993" height="3057" alt="MedFlow_arch_1" src="https://github.com/user-attachments/assets/a70f89c3-76d1-468a-8ceb-888c573b2075" />


## Data Approach

**Synthetic Healthcare Data Generation**
- 100 hospitals across 5-6 regions
- 6 months of historical data
- 5 resource types tracked (ventilators, O2, beds, medications, PPE)
- Realistic patterns: seasonal trends, outbreak events, regional imbalances
- 10,000+ resource allocation scenarios

**Why synthetic?** Ensures reproducibility, privacy compliance, and controlled testing scenarios without access to sensitive real-world patient data.
