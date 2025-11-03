"""
Master evaluation script - evaluates all trained models
"""

import sys
from pathlib import Path

# Add ml_core directory to Python path
ml_core_dir = Path(__file__).parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.insert(0, str(ml_core_dir))

parent_dir = ml_core_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from evaluate_demand_models import evaluate_all_demand_models
from evaluate_shortage_model import evaluate_shortage_model


def main():
    """Evaluate all models"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Evaluate demand forecasting models
    print("\n" + "="*80)
    print("PART 1: DEMAND FORECASTING MODELS")
    print("="*80)
    demand_results = evaluate_all_demand_models(save_results=True)
    
    # Evaluate shortage detection model
    print("\n" + "="*80)
    print("PART 2: SHORTAGE DETECTION MODEL")
    print("="*80)
    shortage_result = evaluate_shortage_model(save_results=True)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    
    demand_success = sum(1 for r in demand_results.values() if r['status'] == 'success')
    demand_total = len(demand_results)
    
    print(f"\nDemand Forecasting Models: {demand_success}/{demand_total} successful")
    print(f"Shortage Detection Model: {'✓' if shortage_result['status'] == 'success' else '✗'}")
    
    print("\n✓ All evaluation results saved to ml_core/evaluation_results/")
    print("="*80)


if __name__ == "__main__":
    main()




