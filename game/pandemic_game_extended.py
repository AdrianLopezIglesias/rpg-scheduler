import numpy as np
from collections import Counter
from .pandemic_game import PandemicGame

# Define a maximum number of cities to ensure all vectors have the same size.
# The standard Pandemic board has 48 cities. Let's use a safe number like 50.
MAX_CITIES = 50

class PandemicGameExtended(PandemicGame):
    """Extends the PandemicGame with fixed-size vector state representations."""

    def get_state_as_vector(self):
        """
        Creates a flat numpy array of a fixed size representing the current game state.
        Pads with zeros if the current map has fewer cities than MAX_CITIES.
        """
        player_loc_idx = self.city_to_idx.get(self.player_location, -1)
        hand_colors = Counter(self.map[card]['color'] for card in self.player_hand)
        disease_statuses = {d['color']: d['status'] for d in self.diseases}
        
        actions_until_infection = (4 - (self.actions_taken % 4)) / 4.0
        remaining_deck_ratio = len(self.deck) / self.initial_deck_size if self.initial_deck_size > 0 else 0
        cards_in_deck_by_color = Counter(self.map[card]['color'] for card in self.deck)

        card_value_features = []
        for color in self.all_possible_colors:
            cards_in_hand = hand_colors.get(color, 0)
            cards_in_deck = cards_in_deck_by_color.get(color, 0)
            needed_for_cure = self.cards_for_cure
            value = (cards_in_hand + cards_in_deck) / needed_for_cure if needed_for_cure > 0 else 10.0
            card_value_features.append(value)

        color_card_features = []
        for color in self.all_possible_colors:
            total = self.total_cards_by_color.get(color, 0)
            if total == 0:
                color_card_features.extend([0.0, 0.0])
            else:
                remaining_in_deck = cards_in_deck_by_color.get(color, 0)
                in_hand = hand_colors.get(color, 0)
                discarded = total - remaining_in_deck - in_hand
                color_card_features.append(remaining_in_deck / total)
                color_card_features.append(discarded / total)
        
        norm_cards_for_cure = self.cards_for_cure / 10.0
        global_features = [
            norm_cards_for_cure, 
            actions_until_infection, 
            remaining_deck_ratio
        ] + color_card_features + card_value_features

        # Calculate the size of features for a single node to use for padding
        single_node_feature_size = (len(self.all_possible_colors) * 4) + 5

        node_features_flat = []
        num_actual_cities = len(self.all_cities)
        
        for i in range(MAX_CITIES):
            if i < num_actual_cities:
                city_name = self.idx_to_city[i]
                
                is_player = 1.0 if i == player_loc_idx else 0.0
                has_card = 1.0 if city_name in self.player_hand else 0.0
                has_center = 1.0 if city_name in self.investigation_centers else 0.0
                
                should_have_center = 0.0
                if not has_center and not any(n in self.investigation_centers for n in self.map[city_name]["neighbors"]):
                    should_have_center = 1.0
                
                distances = [self.get_distance(city_name, c) for c in self.investigation_centers] if self.investigation_centers else []
                dist_to_center = min(distances) if distances else num_actual_cities
                center_proximity = dist_to_center / num_actual_cities if num_actual_cities > 0 else 1.0
                
                city_cubes = self.board_state[city_name]["cubes"]
                cube_features = [city_cubes[c] / 3.0 for c in self.all_possible_colors]
                cure_features = [1.0 if disease_statuses[c] in ['cured', 'eradicated'] else 0.0 for c in self.all_possible_colors]
                hand_features = [hand_colors.get(c, 0) / self.cards_for_cure if self.cards_for_cure > 0 else 0.0 for c in self.all_possible_colors]
                eradicated_features = [1.0 if disease_statuses[c] == 'eradicated' else 0.0 for c in self.all_possible_colors]
                
                local_features = [is_player, has_card, has_center, should_have_center, center_proximity]
                
                all_node_feats = cube_features + cure_features + hand_features + eradicated_features + local_features
                node_features_flat.extend(all_node_feats)
            else:
                # Pad with zeros for non-existent cities
                node_features_flat.extend([0.0] * single_node_feature_size)

        final_vector = np.array(node_features_flat + global_features, dtype=np.float32)
        return final_vector

    def get_feature_vector_size(self):
        """Calculates the fixed size of the flat feature vector."""
        # Per-node features
        per_color_features = len(self.all_possible_colors) * 4 # cubes, cures, hand, eradicated
        local_node_features = 5 # is_player, has_card, has_center, should_center, center_prox
        single_node_total = per_color_features + local_node_features
        all_nodes_total = single_node_total * MAX_CITIES

        # Global features
        base_global = 3 # norm_cards_cure, actions_infection, deck_ratio
        color_card_dist_features = len(self.all_possible_colors) * 2
        card_value_features = len(self.all_possible_colors)
        total_global = base_global + color_card_dist_features + card_value_features

        return all_nodes_total + total_global
