function predictSalary() {
  const age = document.getElementById('age').value;
  const gender = document.getElementById('gender').value;
  const education = document.getElementById('education').value;
  const experience = document.getElementById('experience').value;
  const job_role = document.getElementById('job_role').value;
  const location = document.getElementById('location').value;

  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      age: age,
      gender: gender,
      education: education,
      experience: experience,
      job_role: job_role,
      location: location
    })
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('result').innerText = `Predicted Salary: ₹${data.predicted_salary}`;
  })
  .catch(error => {
    console.error('Error:', error);
    document.getElementById('result').innerText = 'Error predicting salary.';
  });
}
