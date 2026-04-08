"""
Visualization of optimal posting strategies for the Viraltest environment.
Shows engagement multipliers, sleep effects, recommended posting windows,
and simulation results for all 61 test scenarios.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from typing import Callable, List, Tuple, Dict, Any

# Environment constants (matching viraltest_environment.py)
CONTENT_ENERGY_COST = {"reel": 0.25, "carousel": 0.20, "story": 0.08, "text_post": 0.06}
BASE_ENGAGEMENT = {"reel": 0.52, "carousel": 0.55, "story": 0.30, "text_post": 0.37}
REACH_MULT = {"reel": 2.25, "carousel": 1.0, "story": 0.5, "text_post": 0.44}
WEEKEND_PENALTY = 0.7
PEAK_DAYS = (1, 2, 3)  # Tue, Wed, Thu

# Sleep constants
SLEEP_OPTIMAL_AWAKE = 14
SLEEP_HALFLIFE_HOURS = 10
SLEEP_MIN_QUALITY = 0.30


def get_hour_multiplier(hour: int, day: int) -> float:
    """Calculate engagement multiplier for given hour and day."""
    is_weekend = day >= 5
    base = WEEKEND_PENALTY if is_weekend else 1.0

    if 12 <= hour < 15 and day in PEAK_DAYS:
        return base * 1.4
    if 9 <= hour < 12:
        return base * 1.3
    if 18 <= hour < 20:
        return base * 1.25
    if 20 <= hour < 23:
        return base * 1.1
    if hour >= 23 or hour < 6:
        return base * 0.5
    return base * 0.8


def get_sleep_factor(hours_since_sleep: int) -> float:
    """Calculate sleep quality factor (exponential decay)."""
    if hours_since_sleep <= SLEEP_OPTIMAL_AWAKE:
        return 1.0
    hours_over = hours_since_sleep - SLEEP_OPTIMAL_AWAKE
    factor = 0.5 ** (hours_over / SLEEP_HALFLIFE_HOURS)
    return max(SLEEP_MIN_QUALITY, factor)


def create_visualizations():
    """Generate all visualization plots."""
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Viraltest Environment - Optimal Posting Strategy Guide', 
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Hour x Day Engagement Heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = list(range(24))
    
    heatmap_data = np.zeros((24, 7))
    for d in range(7):
        for h in range(24):
            heatmap_data[h, d] = get_hour_multiplier(h, d)
    
    im = ax1.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0.3, vmax=1.5)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(days)
    ax1.set_yticks(range(0, 24, 2))
    ax1.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Hour of Day')
    ax1.set_title('Engagement Multiplier by Hour & Day', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Multiplier')
    
    # Highlight peak zones
    for d in PEAK_DAYS:
        rect = Rectangle((d-0.5, 11.5), 1, 3, linewidth=2, 
                         edgecolor='blue', facecolor='none', linestyle='--')
        ax1.add_patch(rect)
    ax1.text(2, 10.5, 'PEAK\nZONE', fontsize=8, color='blue', ha='center', fontweight='bold')

    # 2. Content Type Comparison
    ax2 = fig.add_subplot(2, 2, 2)
    content_types = list(BASE_ENGAGEMENT.keys())
    x = np.arange(len(content_types))
    width = 0.25
    
    base_vals = [BASE_ENGAGEMENT[ct] for ct in content_types]
    reach_vals = [REACH_MULT[ct] for ct in content_types]
    energy_vals = [CONTENT_ENERGY_COST[ct] for ct in content_types]
    
    # Calculate effective engagement (base * reach)
    effective = [BASE_ENGAGEMENT[ct] * REACH_MULT[ct] for ct in content_types]
    
    bars1 = ax2.bar(x - width, base_vals, width, label='Base Engagement', color='steelblue')
    bars2 = ax2.bar(x, reach_vals, width, label='Reach Multiplier', color='seagreen')
    bars3 = ax2.bar(x + width, energy_vals, width, label='Energy Cost', color='coral')
    
    ax2.set_xlabel('Content Type')
    ax2.set_ylabel('Value')
    ax2.set_title('Content Type Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Reel', 'Carousel', 'Story', 'Text Post'])
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add efficiency annotation
    efficiency = [(BASE_ENGAGEMENT[ct] * REACH_MULT[ct]) / CONTENT_ENERGY_COST[ct] 
                  for ct in content_types]
    for i, (ct, eff) in enumerate(zip(content_types, efficiency)):
        ax2.annotate(f'Eff: {eff:.1f}', (i, max(base_vals[i], reach_vals[i], energy_vals[i]) + 0.1),
                    ha='center', fontsize=8, color='purple')

    # 3. Sleep Quality Decay Curve
    ax3 = fig.add_subplot(2, 2, 3)
    hours_awake = np.linspace(0, 40, 200)
    sleep_quality = [get_sleep_factor(int(h)) for h in hours_awake]
    
    ax3.plot(hours_awake, sleep_quality, 'b-', linewidth=2, label='Quality Factor')
    ax3.axvline(x=SLEEP_OPTIMAL_AWAKE, color='green', linestyle='--', 
                label=f'Optimal threshold ({SLEEP_OPTIMAL_AWAKE}h)')
    ax3.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7,
                label='50% quality (24h awake)')
    ax3.axhline(y=SLEEP_MIN_QUALITY, color='red', linestyle=':', alpha=0.7,
                label=f'Floor ({SLEEP_MIN_QUALITY*100:.0f}%)')
    
    # Fill regions
    ax3.fill_between(hours_awake, sleep_quality, alpha=0.3)
    ax3.axvspan(0, SLEEP_OPTIMAL_AWAKE, alpha=0.1, color='green', label='_No fatigue')
    ax3.axvspan(SLEEP_OPTIMAL_AWAKE, 24, alpha=0.1, color='yellow')
    ax3.axvspan(24, 40, alpha=0.1, color='red')
    
    ax3.set_xlabel('Hours Since Sleep')
    ax3.set_ylabel('Quality Multiplier')
    ax3.set_title('Sleep Deprivation Effect (Exponential Decay)', fontweight='bold')
    ax3.set_xlim(0, 40)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # Add annotations
    ax3.annotate('No impact', xy=(7, 1.02), fontsize=9, color='green')
    ax3.annotate('Mild fatigue', xy=(18, 0.85), fontsize=9, color='orange')
    ax3.annotate('Severe', xy=(30, 0.4), fontsize=9, color='red')

    # 4. Optimal Daily Schedule
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Create a 24-hour timeline
    hours_day = np.arange(24)
    
    # Define activity zones
    sleep_zone = [(0, 7)]  # Sleep 0-7
    low_zone = [(7, 9), (21, 24)]  # Low engagement
    medium_zone = [(9, 12), (15, 18), (20, 21)]  # Medium
    peak_zone = [(12, 15), (18, 20)]  # Peak
    
    # Plot colored bands
    for start, end in sleep_zone:
        ax4.axvspan(start, end, alpha=0.3, color='navy', label='Sleep (rest)' if start == 0 else '')
    for start, end in low_zone:
        ax4.axvspan(start, end, alpha=0.3, color='gray', label='Low engagement' if start == 7 else '')
    for start, end in medium_zone:
        ax4.axvspan(start, end, alpha=0.3, color='yellow', label='Medium' if start == 9 else '')
    for start, end in peak_zone:
        ax4.axvspan(start, end, alpha=0.4, color='green', label='Peak hours' if start == 12 else '')
    
    # Plot engagement curve for peak weekday
    engagement_curve = [get_hour_multiplier(h, 2) for h in hours_day]  # Wednesday
    ax4.plot(hours_day, engagement_curve, 'k-', linewidth=2, marker='o', markersize=4)
    
    # Add recommended actions
    actions = [
        (3, 0.3, 'SLEEP', 'white'),
        (10, 1.35, 'POST #1', 'darkgreen'),
        (13, 1.45, 'PEAK POST', 'darkgreen'),
        (19, 1.3, 'POST #2', 'darkgreen'),
        (16, 0.85, 'Rest/Create', 'gray'),
    ]
    for x, y, text, color in actions:
        ax4.annotate(text, (x, y), fontsize=9, fontweight='bold', 
                    color=color, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Engagement Multiplier')
    ax4.set_title('Optimal Daily Schedule (Peak Weekday: Tue-Thu)', fontweight='bold')
    ax4.set_xlim(0, 24)
    ax4.set_ylim(0, 1.6)
    ax4.set_xticks(range(0, 25, 3))
    ax4.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig('optimal_posting_guide.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: optimal_posting_guide.png")
    
    # Create second figure with strategy summary
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Strategy Recommendations', fontsize=14, fontweight='bold')
    
    # Left: Energy vs Posts tradeoff
    ax5 = axes[0]
    posts_per_day = np.arange(0, 6)
    
    # Calculate energy remaining after N posts of each type
    for ct in content_types:
        cost = CONTENT_ENERGY_COST[ct]
        energy_remaining = [max(0, 1.0 - n * cost) for n in posts_per_day]
        ax5.plot(posts_per_day, energy_remaining, '-o', label=ct.replace('_', ' ').title(), linewidth=2)
    
    ax5.axhline(y=0.4, color='orange', linestyle='--', label='Safe threshold')
    ax5.axhline(y=0.2, color='red', linestyle='--', label='Burnout risk')
    
    ax5.set_xlabel('Posts Per Day')
    ax5.set_ylabel('Energy Remaining')
    ax5.set_title('Energy Drain by Content Type', fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(alpha=0.3)
    ax5.set_xlim(0, 5)
    ax5.set_ylim(0, 1.1)
    
    # Right: Effective Engagement Score
    ax6 = axes[1]
    
    # Calculate total effective engagement for different strategies
    strategies = [
        ('2 Reels/day', 2 * BASE_ENGAGEMENT['reel'] * REACH_MULT['reel'], 2 * CONTENT_ENERGY_COST['reel']),
        ('2 Carousels/day', 2 * BASE_ENGAGEMENT['carousel'] * REACH_MULT['carousel'], 2 * CONTENT_ENERGY_COST['carousel']),
        ('1 Reel + 1 Carousel', BASE_ENGAGEMENT['reel'] * REACH_MULT['reel'] + BASE_ENGAGEMENT['carousel'] * REACH_MULT['carousel'], 
         CONTENT_ENERGY_COST['reel'] + CONTENT_ENERGY_COST['carousel']),
        ('3 Stories/day', 3 * BASE_ENGAGEMENT['story'] * REACH_MULT['story'], 3 * CONTENT_ENERGY_COST['story']),
        ('4 Text Posts/day', 4 * BASE_ENGAGEMENT['text_post'] * REACH_MULT['text_post'], 4 * CONTENT_ENERGY_COST['text_post']),
    ]
    
    names = [s[0] for s in strategies]
    engagement = [s[1] for s in strategies]
    energy_cost = [s[2] for s in strategies]
    efficiency = [e/c for e, c in zip(engagement, energy_cost)]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, engagement, width, label='Total Engagement', color='steelblue')
    bars2 = ax6.bar(x + width/2, energy_cost, width, label='Energy Cost', color='coral')
    
    ax6.set_ylabel('Value')
    ax6.set_title('Daily Strategy Comparison', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(names, rotation=15, ha='right')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # Add efficiency labels
    for i, eff in enumerate(efficiency):
        ax6.annotate(f'Eff: {eff:.1f}', (i, max(engagement[i], energy_cost[i]) + 0.1),
                    ha='center', fontsize=9, color='green', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: strategy_comparison.png")
    
    plt.show()


def print_summary():
    """Print text summary of optimal strategies."""
    print("\n" + "="*70)
    print("OPTIMAL POSTING STRATEGY SUMMARY")
    print("="*70)
    
    print("\n📅 BEST DAYS:")
    print("   • Tuesday, Wednesday, Thursday (peak engagement)")
    print("   • Weekend has 30% penalty")
    
    print("\n⏰ BEST HOURS:")
    print("   • 12:00-15:00 on Tue/Wed/Thu (+40% engagement)")
    print("   • 09:00-12:00 any weekday (+30%)")
    print("   • 18:00-20:00 evening (+25%)")
    print("   • AVOID: 23:00-06:00 (-50%)")
    
    print("\n📱 CONTENT TYPES (by reach efficiency):")
    for ct in ['reel', 'carousel', 'text_post', 'story']:
        eff = (BASE_ENGAGEMENT[ct] * REACH_MULT[ct]) / CONTENT_ENERGY_COST[ct]
        print(f"   • {ct.replace('_', ' ').title():12} - "
              f"Reach: {REACH_MULT[ct]:.2f}x, Energy: {CONTENT_ENERGY_COST[ct]:.0%}, "
              f"Efficiency: {eff:.1f}")
    
    print("\n😴 SLEEP SCHEDULE:")
    print(f"   • No quality impact for first {SLEEP_OPTIMAL_AWAKE} hours awake")
    print("   • Quality halves every 10 hours beyond that")
    print("   • At 24h awake: 50% quality")
    print("   • Rest during 23:00-07:00 to maintain quality")
    
    print("\n🎯 RECOMMENDED DAILY ROUTINE:")
    print("   07:00 - Wake up (2h buffer before posting)")
    print("   09:00-12:00 - Post #1 (morning peak)")
    print("   12:00-15:00 - Post #2 (midday peak on Tue-Thu)")
    print("   15:00-18:00 - Rest or create content")
    print("   18:00-20:00 - Optional Post #3 (evening)")
    print("   23:00 - Sleep (rest actions)")
    
    print("\n⚡ ENERGY MANAGEMENT:")
    print("   • Stay above 0.4 energy (quality drops below 0.5)")
    print("   • 2 reels/day = 50% energy (sustainable)")
    print("   • Use content queue for 50% energy discount")
    print("   • Rest recovers 12% energy + 2h sleep credit")
    
    print("\n" + "="*70)


def run_all_scenarios() -> List[Dict[str, Any]]:
    """Run all 61 scenarios and collect results."""
    from server.viraltest_environment import ViraltestEnvironment
    from models import ViraltestAction
    from test_scenarios import SCENARIOS, TASKS
    
    # Import reset functions
    from test_scenarios import (
        _reset_smart_state, _reset_queue_state, _reset_burst_state,
        _reset_tag_explorer_state, _reset_balanced_state, _reset_queue_heavy_state,
        _reset_alternating_state, _reset_content_creator_state, _reset_nap_state
    )
    
    def _reset_all():
        _reset_smart_state()
        _reset_queue_state()
        _reset_burst_state()
        _reset_tag_explorer_state()
        _reset_balanced_state()
        _reset_queue_heavy_state()
        _reset_alternating_state()
        _reset_content_creator_state()
        _reset_nap_state()
    
    SEED = 42
    results = []
    
    for scenario_name, agent_fn, description in SCENARIOS:
        scenario_results = {
            'name': scenario_name,
            'description': description,
            'scores': {},
            'details': {}
        }
        
        for task in TASKS:
            _reset_all()
            env = ViraltestEnvironment()
            obs = env.reset(task=task, seed=SEED)
            
            rewards = []
            actions = []
            min_energy = 1.0
            max_sleep_debt = 0.0
            burned_out = False
            
            for step in range(1, 169):
                action = agent_fn(obs, step)
                obs = env.step(action)
                r = obs.reward if obs.reward is not None else 0.0
                rewards.append(r)
                actions.append(action.action_type)
                min_energy = min(min_energy, obs.creator_energy)
                max_sleep_debt = max(max_sleep_debt, obs.sleep_debt)
                if obs.done and obs.creator_energy <= 0:
                    burned_out = True
                if obs.done:
                    break
            
            score = (obs.metadata or {}).get("grader_score", 0.0)
            action_counts = Counter(actions)
            
            scenario_results['scores'][task] = score
            scenario_results['details'][task] = {
                'steps': len(rewards),
                'burned_out': burned_out,
                'min_energy': min_energy,
                'max_sleep_debt': max_sleep_debt,
                'final_energy': obs.creator_energy,
                'followers': obs.follower_count,
                'follower_delta': obs.follower_count - 10000,
                'engagement_rate': obs.engagement_rate,
                'posts': action_counts.get('post', 0),
                'rests': action_counts.get('rest', 0),
                'creates': action_counts.get('create_content', 0),
                'total_reward': sum(rewards),
            }
        
        results.append(scenario_results)
    
    return results


def create_scenario_visualizations(results: List[Dict[str, Any]]):
    """Create visualizations for all scenario results."""
    
    # Extract data
    names = [r['name'].replace('SCENARIO ', 'S') for r in results]
    short_names = [n.split(':')[0] for n in names]  # Just "S1", "S2", etc.
    
    engage_scores = [r['scores']['weekly_engage'] for r in results]
    strategic_scores = [r['scores']['weekly_strategic'] for r in results]
    competitive_scores = [r['scores']['weekly_competitive'] for r in results]
    
    # Figure 1: Score comparison bar chart
    fig1, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig1.suptitle('All 61 Scenarios - Performance Scores by Task', fontsize=14, fontweight='bold')
    
    x = np.arange(len(results))
    
    # Color based on score
    def get_colors(scores):
        colors = []
        for s in scores:
            if s >= 0.7:
                colors.append('green')
            elif s >= 0.4:
                colors.append('orange')
            elif s > 0:
                colors.append('coral')
            else:
                colors.append('red')
        return colors
    
    # Weekly Engage
    ax1 = axes[0]
    ax1.bar(x, engage_scores, color=get_colors(engage_scores), edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Score')
    ax1.set_title('Weekly Engage Task', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7+)')
    ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Medium (0.4+)')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Weekly Strategic
    ax2 = axes[1]
    ax2.bar(x, strategic_scores, color=get_colors(strategic_scores), edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Score')
    ax2.set_title('Weekly Strategic Task', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Weekly Competitive
    ax3 = axes[2]
    ax3.bar(x, competitive_scores, color=get_colors(competitive_scores), edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Score')
    ax3.set_title('Weekly Competitive Task', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5)
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('scenario_scores.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: scenario_scores.png")
    
    # Figure 2: Top 15 scenarios
    fig2, ax = plt.subplots(figsize=(14, 8))
    fig2.suptitle('Top 15 Scenarios by Average Score', fontsize=14, fontweight='bold')
    
    # Calculate average scores
    avg_scores = [(r['name'], 
                   (r['scores']['weekly_engage'] + r['scores']['weekly_strategic'] + r['scores']['weekly_competitive']) / 3,
                   r['scores']['weekly_engage'],
                   r['scores']['weekly_strategic'],
                   r['scores']['weekly_competitive'])
                  for r in results]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    top15 = avg_scores[:15]
    
    y = np.arange(len(top15))
    width = 0.25
    
    names_top = [t[0].replace('SCENARIO ', '').split(':')[1].strip()[:25] for t in top15]
    engage_top = [t[2] for t in top15]
    strategic_top = [t[3] for t in top15]
    competitive_top = [t[4] for t in top15]
    
    bars1 = ax.barh(y + width, engage_top, width, label='Engage', color='steelblue')
    bars2 = ax.barh(y, strategic_top, width, label='Strategic', color='seagreen')
    bars3 = ax.barh(y - width, competitive_top, width, label='Competitive', color='coral')
    
    ax.set_xlabel('Score')
    ax.set_ylabel('Scenario')
    ax.set_yticks(y)
    ax.set_yticklabels(names_top)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add average score labels
    for i, (name, avg, e, s, c) in enumerate(top15):
        ax.text(1.02, i, f'Avg: {avg:.2f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    plt.savefig('top_scenarios.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: top_scenarios.png")
    
    # Figure 3: Sleep-related scenarios comparison
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('Sleep-Related Scenarios Analysis', fontsize=14, fontweight='bold')
    
    sleep_scenarios = [r for r in results if any(kw in r['name'].lower() or kw in r['description'].lower() 
                                                  for kw in ['sleep', 'night', 'rest', 'marathon', 'nap'])]
    
    # Left: Scores comparison
    ax_left = axes[0]
    sleep_names = [r['name'].replace('SCENARIO ', '').split(':')[1].strip()[:20] for r in sleep_scenarios]
    sleep_engage = [r['scores']['weekly_engage'] for r in sleep_scenarios]
    sleep_strategic = [r['scores']['weekly_strategic'] for r in sleep_scenarios]
    sleep_competitive = [r['scores']['weekly_competitive'] for r in sleep_scenarios]
    
    y = np.arange(len(sleep_scenarios))
    width = 0.25
    
    ax_left.barh(y + width, sleep_engage, width, label='Engage', color='steelblue')
    ax_left.barh(y, sleep_strategic, width, label='Strategic', color='seagreen')
    ax_left.barh(y - width, sleep_competitive, width, label='Competitive', color='coral')
    
    ax_left.set_xlabel('Score')
    ax_left.set_ylabel('Scenario')
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(sleep_names)
    ax_left.legend(loc='lower right')
    ax_left.set_xlim(0, 1.1)
    ax_left.grid(axis='x', alpha=0.3)
    ax_left.set_title('Sleep Scenario Scores')
    ax_left.invert_yaxis()
    
    # Right: Sleep debt and energy analysis
    ax_right = axes[1]
    
    # Get sleep debt and min energy for strategic task
    sleep_debt_vals = [r['details']['weekly_strategic']['max_sleep_debt'] for r in sleep_scenarios]
    min_energy_vals = [r['details']['weekly_strategic']['min_energy'] for r in sleep_scenarios]
    burned_out = [r['details']['weekly_strategic']['burned_out'] for r in sleep_scenarios]
    
    x = np.arange(len(sleep_scenarios))
    width = 0.35
    
    bars1 = ax_right.bar(x - width/2, sleep_debt_vals, width, label='Max Sleep Debt', color='purple', alpha=0.7)
    bars2 = ax_right.bar(x + width/2, min_energy_vals, width, label='Min Energy', color='orange', alpha=0.7)
    
    # Mark burned out scenarios
    for i, bo in enumerate(burned_out):
        if bo:
            ax_right.annotate('💀', (i, max(sleep_debt_vals[i], min_energy_vals[i]) + 0.05),
                            ha='center', fontsize=12)
    
    ax_right.set_xlabel('Scenario')
    ax_right.set_ylabel('Value')
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(sleep_names, rotation=45, ha='right', fontsize=8)
    ax_right.legend(loc='upper right')
    ax_right.set_ylim(0, 1.2)
    ax_right.grid(axis='y', alpha=0.3)
    ax_right.set_title('Sleep Debt vs Min Energy (💀 = burned out)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('sleep_scenarios.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: sleep_scenarios.png")
    
    # Figure 4: Scenario categories heatmap
    fig4, ax = plt.subplots(figsize=(16, 10))
    fig4.suptitle('All Scenarios - Score Heatmap', fontsize=14, fontweight='bold')
    
    # Create matrix for heatmap
    all_scores = np.array([[r['scores']['weekly_engage'], 
                            r['scores']['weekly_strategic'], 
                            r['scores']['weekly_competitive']] for r in results])
    
    im = ax.imshow(all_scores, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r['name'].replace('SCENARIO ', '') for r in results], fontsize=7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Engage', 'Strategic', 'Competitive'])
    
    # Add score text
    for i in range(len(results)):
        for j in range(3):
            score = all_scores[i, j]
            color = 'white' if score < 0.5 else 'black'
            ax.text(j, i, f'{score:.2f}', ha='center', va='center', fontsize=6, color=color)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('scenario_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: scenario_heatmap.png")
    
    # Figure 5: Action distribution for top performers
    fig5, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig5.suptitle('Action Distribution - Top 6 Strategies', fontsize=14, fontweight='bold')
    
    top6 = avg_scores[:6]
    top6_results = [r for r in results if r['name'] in [t[0] for t in top6]]
    top6_results.sort(key=lambda r: next(t[1] for t in top6 if t[0] == r['name']), reverse=True)
    
    for idx, r in enumerate(top6_results):
        ax = axes[idx // 3, idx % 3]
        details = r['details']['weekly_strategic']
        
        actions = ['Posts', 'Rests', 'Creates']
        counts = [details['posts'], details['rests'], details['creates']]
        colors = ['steelblue', 'seagreen', 'coral']
        
        wedges, texts, autotexts = ax.pie(counts, labels=actions, autopct='%1.0f%%',
                                          colors=colors, startangle=90)
        ax.set_title(r['name'].replace('SCENARIO ', '').split(':')[1].strip()[:25], fontsize=10)
        
        # Add stats
        avg = (r['scores']['weekly_engage'] + r['scores']['weekly_strategic'] + r['scores']['weekly_competitive']) / 3
        ax.text(0, -1.3, f"Avg Score: {avg:.2f} | Energy: {details['final_energy']:.2f}", 
               ha='center', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig('top_actions.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: top_actions.png")
    
    return results


def print_scenario_summary(results: List[Dict[str, Any]]):
    """Print summary table of all scenarios."""
    print("\n" + "="*100)
    print("ALL 61 SCENARIOS - SIMULATION RESULTS")
    print("="*100)
    
    # Calculate averages and sort
    scored_results = []
    for r in results:
        avg = (r['scores']['weekly_engage'] + r['scores']['weekly_strategic'] + r['scores']['weekly_competitive']) / 3
        scored_results.append((r, avg))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<5} {'Scenario':<45} {'Engage':>8} {'Strategic':>10} {'Competitive':>12} {'Avg':>8}")
    print("-" * 100)
    
    for rank, (r, avg) in enumerate(scored_results, 1):
        name = r['name'].replace('SCENARIO ', '')[:43]
        e = r['scores']['weekly_engage']
        s = r['scores']['weekly_strategic']
        c = r['scores']['weekly_competitive']
        
        # Add indicator for top performers
        indicator = "🏆" if rank <= 3 else "⭐" if rank <= 10 else "  "
        print(f"{indicator}{rank:<3} {name:<45} {e:>8.4f} {s:>10.4f} {c:>12.4f} {avg:>8.4f}")
    
    print("\n" + "="*100)
    print("TOP 10 DETAILED ANALYSIS")
    print("="*100)
    
    for rank, (r, avg) in enumerate(scored_results[:10], 1):
        print(f"\n#{rank} {r['name']}")
        print(f"   Description: {r['description']}")
        
        for task in ['weekly_engage', 'weekly_strategic', 'weekly_competitive']:
            d = r['details'][task]
            print(f"   {task}: Score={r['scores'][task]:.4f} | "
                  f"Posts={d['posts']} Rests={d['rests']} Creates={d['creates']} | "
                  f"Energy={d['final_energy']:.2f} | Followers={d['follower_delta']:+d}")
    
    # Sleep scenario analysis
    print("\n" + "="*100)
    print("SLEEP MECHANICS ANALYSIS")
    print("="*100)
    
    sleep_keywords = ['sleep', 'night', 'rest', 'marathon', 'nap', 'awake']
    sleep_results = [(r, (r['scores']['weekly_engage'] + r['scores']['weekly_strategic'] + r['scores']['weekly_competitive']) / 3) 
                     for r in results 
                     if any(kw in r['name'].lower() or kw in r['description'].lower() for kw in sleep_keywords)]
    sleep_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Scenario':<40} {'Avg Score':>10} {'Max Sleep Debt':>15} {'Burned Out':>12}")
    print("-" * 80)
    
    for r, avg in sleep_results:
        name = r['name'].replace('SCENARIO ', '').split(':')[1].strip()[:38]
        debt = r['details']['weekly_strategic']['max_sleep_debt']
        bo = "YES 💀" if r['details']['weekly_strategic']['burned_out'] else "No"
        print(f"{name:<40} {avg:>10.4f} {debt:>15.3f} {bo:>12}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    print("Generating optimal posting visualizations...")
    print_summary()
    create_visualizations()
    
    print("\n" + "="*70)
    print("Running all 61 scenarios...")
    print("="*70)
    
    results = run_all_scenarios()
    print_scenario_summary(results)
    create_scenario_visualizations(results)
    
    print("\n✅ All visualizations generated!")
    print("   - optimal_posting_guide.png")
    print("   - strategy_comparison.png")
    print("   - scenario_scores.png")
    print("   - top_scenarios.png")
    print("   - sleep_scenarios.png")
    print("   - scenario_heatmap.png")
    print("   - top_actions.png")
