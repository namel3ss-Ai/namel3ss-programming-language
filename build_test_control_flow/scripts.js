// Chart for chart_features_0
var ctx_chart_features_0 = document.getElementById('chart_features_0').getContext('2d');
new Chart(ctx_chart_features_0, {
  type: 'bar',
  data: {
    labels: ["A", "B", "C", "D", "E"],
    datasets: [{
      label: 'Analytics',
      data: [45, 16, 15, 79, 60],
      backgroundColor: 'rgba(54, 162, 235, 0.6)'
    }]
  },
  options: { responsive: true }
});
