import plotly.graph_objects as go

science_template = {
      'layout': go.Layout({
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'colorscale': {
                  'diverging': [
                        [0  , '#8e0152'],
                        [0.1, '#c51b7d'],
                        [0.2, '#de77ae'],
                        [0.3, '#f1b6da'],
                        [0.4, '#fde0ef'],
                        [0.5, '#f7f7f7'],
                        [0.6, '#e6f5d0'],
                        [0.7, '#b8e186'],
                        [0.8, '#7fbc41'],
                        [0.9, '#4d9221'],
                        [1, '#276419']
                  ],
                  'sequential': [
                        [0.0               , '#0d0887'],
                        [0.1111111111111111, '#46039f'],
                        [0.2222222222222222, '#7201a8'],
                        [0.3333333333333333, '#9c179e'],
                        [0.4444444444444444, '#bd3786'], 
                        [0.5555555555555556, '#d8576b'],
                        [0.6666666666666666, '#ed7953'],
                        [0.7777777777777778, '#fb9f3a'],
                        [0.8888888888888888, '#fdca26'],
                        [1.0,                '#f0f921']
                  ],
                  'sequentialminus': [
                        [0.0,                '#0d0887'],
                        [0.1111111111111111, '#46039f'],
                        [0.2222222222222222, '#7201a8'],
                        [0.3333333333333333, '#9c179e'],
                        [0.4444444444444444, '#bd3786'],
                        [0.5555555555555556, '#d8576b'],
                        [0.6666666666666666, '#ed7953'],
                        [0.7777777777777778, '#fb9f3a'],
                        [0.8888888888888888, '#fdca26'],
                        [1.0,                '#f0f921']
                  ]
            },
            # Those are the colors for the lines etx
            'colorway': [
                  '#636efa',
                  '#EF553B',
                  '#00cc96',
                  '#ab63fa',
                  '#FFA15A',
                  '#19d3f3',
                  '#FF6692',
                  '#B6E880',
                  '#FF97FF',
                  '#FECB52'
            ],
            'xaxis': {
                  'ticks': 'inside',
                  'color': '#444444',
                  'showline': True,
                  'mirror': 'all',
                  'exponentformat': 'power',
            },
            'yaxis': {
                  'ticks': 'inside',
                  'color': '#444444',
                  'showline': True,
                  'mirror': 'all',
                  'exponentformat': 'power'
            },
            'legend': {
                  'bgcolor': 'rgba(0,0,0,0)',
                  'orientation': "v",
                  'yanchor': "auto",
                  'y': 1,
                  'xanchor': "left",
                  'x': 1.01,
            }
      }),
      'data': {
            'scatter': [{
                  'textposition': 'top center'
            }]
      }
}
