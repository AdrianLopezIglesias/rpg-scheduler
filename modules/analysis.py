import numpy as np
from .utils import log

def analyze_generation_data(list_of_games):
    """Analyzes game data and returns a dictionary of performance metrics."""
    num_games = len(list_of_games)
    winning_games = [g for g in list_of_games if g.get("final_result") == "win"]
    win_rate = len(winning_games) / num_games * 100 if num_games > 0 else 0
    
    if winning_games:
        win_lengths = [g["total_actions"] for g in winning_games]
        avg_win_speed = np.mean(win_lengths)
        fastest_win = min(win_lengths)
    else:
        avg_win_speed = "N/A"
        fastest_win = "N/A"

    analysis = {
        "win_rate_percent": win_rate,
        "avg_actions_to_win": avg_win_speed,
        "fastest_win_actions": fastest_win
    }

    avg_win_speed_formatted = f"{analysis['avg_actions_to_win']:.2f}" if isinstance(analysis['avg_actions_to_win'], float) else "N/A"
    log(f"--- Analysis --- Win Rate: {analysis['win_rate_percent']:.2f}% | Avg. Win Speed: {avg_win_speed_formatted} | Fastest Win: {analysis['fastest_win_actions']} actions")
    return analysis