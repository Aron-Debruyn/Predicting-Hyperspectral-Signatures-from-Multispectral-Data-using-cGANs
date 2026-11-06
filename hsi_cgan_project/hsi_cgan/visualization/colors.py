"""Colour maps and the colour lookup shared by the loss/validation plots.

Contents
--------
new_ordered_scenario_keys / new_scenario_color_map  Keys and colours of the 16 two-band scenarios.
ordered_scenario_keys / scenario_color_map          Keys and colours of the six main band scenarios.
get_color_for_scenario                              Pick the colour whose key occurs in a model name.
"""



new_ordered_scenario_keys = [
    'B-RE717', 'R-RE717', 'G-RE717',
    'G-NIR', 'B-NIR', 'R-NIR',
    'G531-NIR', 'R650-NIR', 'R-RE705',
    'R-RE740', 'B-RE705', 'B-RE740',
    'G-RE705', 'G-RE740', 'G531-RE', 'R650-RE'
]


# 16 new scenarios
new_scenario_color_map = {
    'B-RE717': '#1f77b4', 'R-RE717': '#ff7f0e', 'G-RE717': '#2ca02c',
    'G-NIR': '#d62728', 'B-NIR': '#9467bd', 'R-NIR': '#8c564b',
    'G531-NIR': '#e377c2', 'R650-NIR': '#7f7f7f', 'R-RE705': '#bcbd22',
    'R-RE740': '#17becf', 'B-RE705': '#aec7e8', 'B-RE740': '#ffbb78',
    'G-RE705': '#98df8a', 'G-RE740': '#ff9896', 'G531-RE': '#c5b0d5',
    'R650-RE': '#c49c94', 'default': 'gray'
}


ordered_scenario_keys = [
    'RGB + NIR',
    'Centers + Bandwidth',
    'RGB',
    'PCA',
    'MI',
    'Centers'
]


scenario_color_map = {
    'RGB + NIR': 'red',
    'RGB': 'blue',
    'PCA': 'green',
    'MI': 'purple',
    'Centers': 'orange',
    'Centers + Bandwidth': 'deepskyblue',
    'default': 'gray'
}


def get_color_for_scenario(model_name, color_map, ordered_keys):
    for scenario_key in ordered_keys:
        if scenario_key in model_name:
            return color_map.get(scenario_key, color_map['default'])
    return color_map['default']
