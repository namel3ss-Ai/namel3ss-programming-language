// Chart for chart_index_0
var ctx_chart_index_0 = document.getElementById('chart_index_0').getContext('2d');
new Chart(ctx_chart_index_0, {
  type: 'line',
  data: {
    labels: ["A", "B", "C", "D", "E"],
    datasets: [{
      label: 'Revenue Growth',
      data: [99, 79, 19, 96, 16],
      backgroundColor: 'var(--primary)'
    }]
  },
  options: { responsive: true }
});

document.getElementById('form_feedback_0').addEventListener('submit', function(e) {
  e.preventDefault();
  showToast("Thank you for your feedback!");
});

document.getElementById('action_btn_admin_0').addEventListener('click', function() {
  console.log('Update orders: status = "APPROVED" where ');
  showToast('Updated orders');
  showToast("Order approved");
  window.location.href = 'admin.html';
});
