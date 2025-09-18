import json
import os
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def load_generation_summary(summary_file):
    """Load the generation summary JSON file"""
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Generation summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def filter_relevant_runs(generation_data):
    """Filter runs for dreambooth_only and animatediff_and_dreambooth modes with success=true"""
    relevant_modes = ["dreambooth_only", "animatediff_and_dreambooth"]
    relevant_runs = []
    
    for run in generation_data.get("runs", []):
        if (run.get("generation_mode") in relevant_modes and 
            run.get("success") == True):
            relevant_runs.append(run)
    
    return relevant_runs

def run_prompt_similarity_evaluation(stamp, method="both"):
    """Run the prompt similarity evaluation for a given stamp"""
    print(f"\n{'='*60}")
    print(f"RUNNING EVALUATION FOR: {stamp}")
    print(f"{'='*60}")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_similarity_script = os.path.join(script_dir, "prompt_similarity.py")
    
    if not os.path.exists(prompt_similarity_script):
        raise FileNotFoundError(f"prompt_similarity.py not found at: {prompt_similarity_script}")
    
    try:
        # Run the prompt similarity script
        cmd = [sys.executable, prompt_similarity_script, "--stamp", stamp, "--method", method]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode != 0:
            print(f"Error running evaluation for {stamp}:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        print(f"Successfully completed evaluation for {stamp}")
        
        # Load the results
        results_file = f"/graphics/scratch2/students/webereli/apt/logs/{stamp}/prompt_similarity_results.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Results file not found at {results_file}")
            return None
            
    except Exception as e:
        print(f"Exception running evaluation for {stamp}: {e}")
        return None

def extract_key_findings(evaluation_results, run_info):
    """Extract key findings from evaluation results"""
    if not evaluation_results or "evaluation_summary" not in evaluation_results:
        return None
    
    key_findings = evaluation_results["evaluation_summary"].get("key_findings", {})
    
    # Add run metadata to key findings
    enhanced_findings = {
        "run_info": {
            "name": run_info.get("name"),
            "generation_mode": run_info.get("generation_mode"),
            "stamp": run_info.get("stamp"),
            "character": run_info.get("character", "N/A"),
            "description": run_info.get("description")
        },
        "key_findings": key_findings
    }
    
    return enhanced_findings

def create_combined_summary(all_findings, generation_data, output_dir):
    """Create a combined summary of all key findings"""
    
    # Group findings by generation mode
    grouped_findings = {
        "dreambooth_only": [],
        "animatediff_and_dreambooth": []
    }
    
    for finding in all_findings:
        if finding:
            mode = finding["run_info"]["generation_mode"]
            if mode in grouped_findings:
                grouped_findings[mode].append(finding)
    
    # Calculate averages by mode
    mode_averages = {}
    for mode, findings in grouped_findings.items():
        if not findings:
            continue
            
        all_frames_scores = {
            "original": [],
            "cleaned": [],
            "differences": []
        }
        median_scores = {
            "original": [],
            "cleaned": [],
            "differences": []
        }
        
        valid_runs = 0
        for finding in findings:
            kf = finding.get("key_findings", {})
            
            # All frames method
            all_frames = kf.get("all_frames_method", {})
            if all_frames.get("original_average_clip_score") is not None:
                all_frames_scores["original"].append(all_frames["original_average_clip_score"])
                all_frames_scores["cleaned"].append(all_frames["cleaned_average_clip_score"])
                all_frames_scores["differences"].append(all_frames["average_score_difference"])
            
            # Median frame method
            median_frames = kf.get("median_frame_method", {})
            if median_frames.get("original_average_clip_score") is not None:
                median_scores["original"].append(median_frames["original_average_clip_score"])
                median_scores["cleaned"].append(median_frames["cleaned_average_clip_score"])
                median_scores["differences"].append(median_frames["average_score_difference"])
                valid_runs += 1
        
        mode_averages[mode] = {
            "total_runs": len(findings),
            "valid_runs": valid_runs,
            "all_frames_method": {
                "average_original_score": round(sum(all_frames_scores["original"]) / len(all_frames_scores["original"]), 4) if all_frames_scores["original"] else None,
                "average_cleaned_score": round(sum(all_frames_scores["cleaned"]) / len(all_frames_scores["cleaned"]), 4) if all_frames_scores["cleaned"] else None,
                "average_difference": round(sum(all_frames_scores["differences"]) / len(all_frames_scores["differences"]), 4) if all_frames_scores["differences"] else None
            },
            "median_frame_method": {
                "average_original_score": round(sum(median_scores["original"]) / len(median_scores["original"]), 4) if median_scores["original"] else None,
                "average_cleaned_score": round(sum(median_scores["cleaned"]) / len(median_scores["cleaned"]), 4) if median_scores["cleaned"] else None,
                "average_difference": round(sum(median_scores["differences"]) / len(median_scores["differences"]), 4) if median_scores["differences"] else None
            }
        }
    
    # Create comprehensive summary
    combined_summary = {
        "evaluation_metadata": {
            "generated_at": datetime.now().isoformat(),
            "mp3_file": generation_data.get("mp3_file"),
            "total_runs_evaluated": len(all_findings),
            "evaluation_modes": ["dreambooth_only", "animatediff_and_dreambooth"]
        },
        "mode_averages": mode_averages,
        "individual_run_findings": all_findings
    }
    
    # Get mp3_file name for filename prefix
    mp3_file = generation_data.get("mp3_file", "unknown")
    
    # Save combined summary with mp3_file prefix
    output_file = os.path.join(output_dir, f"{mp3_file}_combined_prompt_similarity_summary.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    return combined_summary, output_file

def print_summary_report(combined_summary):
    """Print a formatted summary report to console"""
    print("\n" + "="*80)
    print("COMBINED PROMPT SIMILARITY EVALUATION SUMMARY")
    print("="*80)
    
    metadata = combined_summary["evaluation_metadata"]
    print(f"MP3 File: {metadata['mp3_file']}")
    print(f"Total Runs Evaluated: {metadata['total_runs_evaluated']}")
    print(f"Generated: {metadata['generated_at']}")
    
    mode_averages = combined_summary["mode_averages"]
    
    for mode, averages in mode_averages.items():
        print(f"\n{mode.upper().replace('_', ' ')}:")
        print(f"  Total runs: {averages['total_runs']}")
        print(f"  Valid runs: {averages['valid_runs']}")
        
        if averages["all_frames_method"]["average_original_score"] is not None:
            print("  All Frames Method:")
            print(f"    Average Original Score:  {averages['all_frames_method']['average_original_score']:.4f}")
            print(f"    Average Cleaned Score:   {averages['all_frames_method']['average_cleaned_score']:.4f}")
            print(f"    Average Difference:      {averages['all_frames_method']['average_difference']:.4f}")
        
        if averages["median_frame_method"]["average_original_score"] is not None:
            print("  Median Frame Method:")
            print(f"    Average Original Score:  {averages['median_frame_method']['average_original_score']:.4f}")
            print(f"    Average Cleaned Score:   {averages['median_frame_method']['average_cleaned_score']:.4f}")
            print(f"    Average Difference:      {averages['median_frame_method']['average_difference']:.4f}")
    
    print("\nINDIVIDUAL RUN RESULTS:")
    print("-" * 40)
    
    for finding in combined_summary["individual_run_findings"]:
        if finding:
            run_info = finding["run_info"]
            kf = finding["key_findings"]
            
            print(f"\n{run_info['name']} ({run_info['generation_mode']}):")
            print(f"  Character: {run_info['character']}")
            print(f"  Stamp: {run_info['stamp']}")
            
            # All frames results
            all_frames = kf.get("all_frames_method", {})
            if all_frames.get("original_average_clip_score") is not None:
                print(f"  All Frames - Original: {all_frames['original_average_clip_score']:.4f}, Cleaned: {all_frames['cleaned_average_clip_score']:.4f}, Diff: {all_frames['average_score_difference']:.4f}")
            
            # Median frame results
            median_frames = kf.get("median_frame_method", {})
            if median_frames.get("original_average_clip_score") is not None:
                print(f"  Median Frame - Original: {median_frames['original_average_clip_score']:.4f}, Cleaned: {median_frames['cleaned_average_clip_score']:.4f}, Diff: {median_frames['average_score_difference']:.4f}")

def main():
    """Main function to run prompt similarity evaluation for all relevant runs"""
    parser = argparse.ArgumentParser(description='Run prompt similarity evaluation for all relevant generation runs')
    parser.add_argument('--summary-file', required=True, help='Path to generation summary JSON file (e.g., apt_generation_summary.json)')
    parser.add_argument('--method', choices=['all', 'median', 'both'], default='both',
                      help='Method to use for evaluation: all (all frames), median (median frame), or both')
    parser.add_argument('--output-dir', help='Output directory for combined summary (default: same directory as summary file)')
    
    args = parser.parse_args()
    
    try:
        # Load generation summary
        print(f"Loading generation summary from: {args.summary_file}")
        generation_data = load_generation_summary(args.summary_file)
        
        # Filter relevant runs
        relevant_runs = filter_relevant_runs(generation_data)
        print(f"Found {len(relevant_runs)} relevant runs (dreambooth_only or animatediff_and_dreambooth with success=true)")
        
        if not relevant_runs:
            print("No relevant runs found. Exiting.")
            return
        
        # Set output directory
        output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.summary_file))
        
        # Run evaluation for each relevant run
        all_findings = []
        successful_runs = 0
        
        for i, run in enumerate(relevant_runs, 1):
            stamp = run["stamp"]
            print(f"\nProcessing run {i}/{len(relevant_runs)}: {run['name']} ({stamp})")
            
            # Run the evaluation
            evaluation_results = run_prompt_similarity_evaluation(stamp, args.method)
            
            if evaluation_results:
                # Extract key findings
                key_findings = extract_key_findings(evaluation_results, run)
                all_findings.append(key_findings)
                successful_runs += 1
            else:
                print(f"Failed to get results for {stamp}")
                all_findings.append(None)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETED")
        print(f"{'='*60}")
        print(f"Total runs processed: {len(relevant_runs)}")
        print(f"Successful evaluations: {successful_runs}")
        print(f"Failed evaluations: {len(relevant_runs) - successful_runs}")
        
        # Create combined summary
        if successful_runs > 0:
            print(f"\nCreating combined summary...")
            combined_summary, summary_file = create_combined_summary(all_findings, generation_data, output_dir)
            
            print(f"Combined summary saved to: {summary_file}")
            
            # Print formatted report
            print_summary_report(combined_summary)
            
        else:
            print("No successful evaluations to summarize.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()