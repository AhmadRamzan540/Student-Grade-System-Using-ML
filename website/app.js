// ============================================================
// EduGrade - Student Performance Predictor | App Logic
// ============================================================

// ---- NAVBAR SCROLL EFFECT ----
const navbar = document.getElementById('navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 10);
  });
}

// ---- HAMBURGER MENU ----
const hamburger = document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');
if (hamburger && navLinks) {
  hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
  document.addEventListener('click', (e) => {
    if (!hamburger.contains(e.target) && !navLinks.contains(e.target)) {
      navLinks.classList.remove('open');
    }
  });
}

// ---- ACTIVE NAV LINK ----
function updateActiveNavLink() {
  const path = window.location.pathname;
  const filename = path.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href === filename || (filename === '' && href === 'index.html')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}
// Run on load
document.addEventListener('DOMContentLoaded', updateActiveNavLink);
updateActiveNavLink();

// ---- INTERSECTION OBSERVER FOR ANIMATIONS ----
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const el = entry.target;
      const delay = parseInt(el.dataset.delay || 0);
      setTimeout(() => el.classList.add('visible'), delay);
      observer.unobserve(el);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card, .accuracy-card, .about-stat-card').forEach(el => observer.observe(el));

// ---- ACCURACY BAR ANIMATION ----
const accBarObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.querySelectorAll('.acc-bar').forEach(bar => {
        const width = getComputedStyle(bar).getPropertyValue('--width').trim();
        setTimeout(() => { bar.style.width = width; }, 200);
      });
      accBarObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.2 });

document.querySelectorAll('.accuracy-section').forEach(el => accBarObserver.observe(el));

// ---- MATH SCORE SLIDER <-> INPUT SYNC ----
const mathInput = document.getElementById('mathScore');
const mathSlider = document.getElementById('mathSlider');
const scoreLabel = document.getElementById('scoreLabel');
const scoreIndicator = document.getElementById('scoreIndicator');

function updateScoreUI(val) {
  if (!scoreLabel || !scoreIndicator) return;
  const num = parseInt(val);
  if (isNaN(num) || num < 0 || num > 100) {
    scoreLabel.textContent = '–';
    scoreIndicator.style.background = '';
    scoreIndicator.style.color = '';
    scoreIndicator.style.borderColor = '';
    return;
  }
  scoreLabel.textContent = `${num}/100`;
  if (num >= 70) {
    scoreIndicator.style.background = '#d1fae5';
    scoreIndicator.style.color = '#059669';
    scoreIndicator.style.borderColor = '#a7f3d0';
  } else if (num >= 50) {
    scoreIndicator.style.background = '#fef3c7';
    scoreIndicator.style.color = '#d97706';
    scoreIndicator.style.borderColor = '#fde68a';
  } else {
    scoreIndicator.style.background = '#fee2e2';
    scoreIndicator.style.color = '#dc2626';
    scoreIndicator.style.borderColor = '#fca5a5';
  }
}

if (mathInput && mathSlider) {
  mathInput.addEventListener('input', () => {
    const val = mathInput.value;
    if (val !== '' && val >= 0 && val <= 100) {
      mathSlider.value = val;
    }
    updateScoreUI(val);
  });

  mathSlider.addEventListener('input', () => {
    mathInput.value = mathSlider.value;
    updateScoreUI(mathSlider.value);
  });

  updateScoreUI(mathSlider.value);
}

// ---- PREDICTION ENGINE ----
// Approximates the trained Decision Tree + ensemble with demographic adjustments
function predictGrade({ gender, race, education, lunch, prep, mathScore }) {
  const score = parseInt(mathScore);

  // --- Primary signal: math score (sigmoid-like around threshold 50)
  // Decision tree splits around 50 as pass/fail boundary
  let baseProbPass;
  if (score >= 80) baseProbPass = 0.97;
  else if (score >= 70) baseProbPass = 0.94;
  else if (score >= 65) baseProbPass = 0.91;
  else if (score >= 60) baseProbPass = 0.88;
  else if (score >= 55) baseProbPass = 0.83;
  else if (score >= 50) baseProbPass = 0.74;
  else if (score >= 45) baseProbPass = 0.55;
  else if (score >= 40) baseProbPass = 0.38;
  else if (score >= 35) baseProbPass = 0.25;
  else if (score >= 25) baseProbPass = 0.14;
  else baseProbPass = 0.07;

  // --- Adjustments from demographic / socioeconomic features ---
  let delta = 0;

  // Lunch: standard is a proxy for higher income, better resources
  if (lunch === 'standard') delta += 0.04;
  else delta -= 0.04;

  // Test prep course: completing it has a strong positive impact
  if (prep === 'completed') delta += 0.06;
  else delta -= 0.02;

  // Parental education level
  const eduMap = {
    'some high school': -0.03,
    'high school': -0.01,
    'some college': 0.01,
    "associate's degree": 0.02,
    "bachelor's degree": 0.03,
    "master's degree": 0.04
  };
  delta += eduMap[education] ?? 0;

  // Race/ethnicity adjustment (based on dataset distribution patterns)
  const raceMap = { 'A': -0.02, 'B': -0.01, 'C': 0.0, 'D': 0.01, 'E': 0.02 };
  delta += raceMap[race] ?? 0;

  // Gender adjustment (slight bias observed in dataset)
  if (gender === 'female') delta += 0.01;

  // Clamp to valid probability range
  let prob = Math.max(0.04, Math.min(0.97, baseProbPass + delta));

  const pass = prob >= 0.50;
  const rawConfidence = pass ? prob : 1 - prob;
  // Scale confidence relative to model accuracy (96.0%)
  // Maps range [0.5, 1.0] to [0.5, 0.96]
  const calibratedConfidence = 0.5 + (rawConfidence - 0.5) * 0.92;
  return { pass, confidence: calibratedConfidence, rawProb: prob };
}

// ---- FORM VALIDATION ----
function validateForm() {
  let valid = true;

  const fields = [
    { name: 'gender',    errorId: 'gender-error',    groupId: 'gender-group',  type: 'radio' },
    { name: 'race',      errorId: 'race-error',       groupId: null,            type: 'select', id: 'race' },
    { name: 'education', errorId: 'education-error',  groupId: null,            type: 'select', id: 'education' },
    { name: 'lunch',     errorId: 'lunch-error',      groupId: 'lunch-group',   type: 'radio' },
    { name: 'prep',      errorId: 'prep-error',       groupId: 'prep-group',    type: 'radio' },
  ];

  fields.forEach(f => {
    const err = document.getElementById(f.errorId);
    if (!err) return;
    let hasValue = false;
    if (f.type === 'radio') {
      hasValue = !!document.querySelector(`input[name="${f.name}"]:checked`);
    } else {
      hasValue = !!document.getElementById(f.id)?.value;
    }
    err.classList.toggle('visible', !hasValue);
    if (!hasValue) valid = false;
  });

  // Math score validation
  const mathErr = document.getElementById('math-error');
  if (mathErr && mathInput) {
    const mathVal = parseInt(mathInput.value);
    const mathValid = !isNaN(mathVal) && mathVal >= 0 && mathVal <= 100;
    mathErr.classList.toggle('visible', !mathValid);
    if (!mathValid) valid = false;
  } else {
    valid = false;
  }

  return valid;
}

// ---- SHOW RESULT ----
function showResult(data) {
  const panel = document.getElementById('resultPanel');
  const placeholder = document.getElementById('resultPlaceholder');
  const content = document.getElementById('resultContent');
  const verdict = document.getElementById('resultVerdict');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictLabel = document.getElementById('verdictLabel');
  const confBar = document.getElementById('confBar');
  const confValue = document.getElementById('confValue');
  const breakdown = document.getElementById('resultBreakdown');
  const tip = document.getElementById('resultTip');

  if (placeholder) placeholder.classList.add('hidden');
  if (content) content.classList.remove('hidden');

  // Update model tag dynamically
  const modelTag = document.querySelector('.result-model-tag');
  if (modelTag) {
    if (data.modelName) {
      modelTag.textContent = `${titleCase(data.modelName.toLowerCase())} Model`;
      modelTag.style.background = '#e0e7ff';
      modelTag.style.color = '#4338ca';
    } else {
      modelTag.textContent = 'Local Simulation Model';
      modelTag.style.background = '#f1f5f9';
      modelTag.style.color = '#475569';
    }
  }

  // Verdict styles
  if (verdict && verdictLabel && verdictIcon) {
    verdict.className = `result-verdict ${data.pass ? 'pass' : 'fail'}`;
    verdictLabel.textContent = data.pass ? 'PASS' : 'FAIL';
    verdictIcon.innerHTML = data.pass
      ? `<svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`
      : `<svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
  }

  // Confidence bar
  if (confBar && confValue) {
    confBar.className = `conf-bar ${data.pass ? 'pass' : 'fail'}`;
    confBar.style.width = '0%';
    confValue.textContent = '0%';
    setTimeout(() => {
      const pct = Math.round(data.confidence * 100);
      confBar.style.width = `${pct}%`;
      animateNumber(confValue, 0, pct, 1200, v => `${v}%`);
    }, 150);
  }

  // Input breakdown
  const gender = document.querySelector('input[name="gender"]:checked')?.value || '';
  const race = document.getElementById('race')?.value || '';
  const edu = document.getElementById('education')?.value || '';
  const lunch = document.querySelector('input[name="lunch"]:checked')?.value || '';
  const prep = document.querySelector('input[name="prep"]:checked')?.value || '';
  const math = mathInput ? mathInput.value : '';

  if (breakdown) {
    breakdown.innerHTML = `
      <div class="breakdown-row"><span class="breakdown-key">Gender</span><span class="breakdown-val">${capitalize(gender)}</span></div>
      <div class="breakdown-row"><span class="breakdown-key">Race / Ethnicity</span><span class="breakdown-val">Group ${race}</span></div>
      <div class="breakdown-row"><span class="breakdown-key">Parental Education</span><span class="breakdown-val">${titleCase(edu)}</span></div>
      <div class="breakdown-row"><span class="breakdown-key">Lunch</span><span class="breakdown-val">${titleCase(lunch)}</span></div>
      <div class="breakdown-row"><span class="breakdown-key">Test Prep Course</span><span class="breakdown-val">${titleCase(prep)}</span></div>
      <div class="breakdown-row"><span class="breakdown-key">Math Score</span><span class="breakdown-val">${math} / 100</span></div>
    `;
  }

  // Contextual tip
  if (tip) {
    const mathNum = parseInt(math) || 0;
    let tipText = '';
    if (data.pass) {
      if (mathNum >= 70) tipText = 'Excellent math performance! The student is well above the passing threshold. Completing the test preparation course would further solidify results.';
      else if (mathNum >= 50) tipText = 'The student passes, but the math score is close to the borderline. Encourage consistent study habits to improve the safety margin.';
      else tipText = 'The student passes due to strong supporting factors. Focusing on improving math scores will lead to even better outcomes.';
    } else {
      if (mathNum < 50) tipText = `The math score of ${mathNum} is below the passing threshold (50). This is the primary reason for the prediction. Targeted math tutoring is recommended.`;
      else if (prep === 'none') tipText = 'Enrolling in the test preparation course can significantly boost performance. It has one of the strongest positive effects in this model.';
      else tipText = 'The combined effect of the demographic factors and scores suggests a high risk of failing. Early intervention and additional academic support is recommended.';
    }
    tip.textContent = tipText;
  }

  // Scroll to result panel
  if (panel) {
    panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// ---- FORM SUBMIT ----
const predictionForm = document.getElementById('predictionForm');
if (predictionForm) {
  predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    const gender = document.querySelector('input[name="gender"]:checked').value;
    const race = document.getElementById('race').value;
    const education = document.getElementById('education').value;
    const lunch = document.querySelector('input[name="lunch"]:checked').value;
    const prep = document.querySelector('input[name="prep"]:checked').value;
    const mathScore = parseInt(mathInput.value);

    // Brief loading feel
    const btn = document.getElementById('predictBtn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="animation:spin 0.8s linear infinite"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-.08-8.62"/></svg> Analyzing…`;
    }

    try {
      // Attempt request to Python ML Flask Server
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          gender,
          race,
          education,
          lunch,
          prep,
          mathScore
        }),
        // Set reasonable short timeout so we fall back quickly if server is offline
        signal: AbortSignal.timeout(1500)
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.success) {
        showResult({
          pass: data.prediction === 1,
          confidence: data.confidence,
          modelName: data.model_name
        });
        console.log(`[ML] Prediction served from active Flask API using ${data.model_name} model.`);
      } else {
        throw new Error(data.error || 'Prediction failed');
      }
    } catch (err) {
      console.warn(`[ML Fallback] Flask API server is offline or errored: ${err.message}. Falling back to local JS simulation.`);
      // Run local JS simulation heuristic as fallback
      const fallbackResult = predictGrade({ gender, race, education, lunch, prep, mathScore });
      showResult(fallbackResult);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Predict Performance`;
      }
    }
  });
}

// ---- RESET BUTTON ----
const resetBtn = document.getElementById('resetBtn');
if (resetBtn) {
  resetBtn.addEventListener('click', () => {
    const resultContent = document.getElementById('resultContent');
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const predictionForm = document.getElementById('predictionForm');
    const predictorSection = document.getElementById('predictor');
    
    if (resultContent) resultContent.classList.add('hidden');
    if (resultPlaceholder) resultPlaceholder.classList.remove('hidden');
    if (predictionForm) predictionForm.reset();
    
    if (mathSlider) {
      mathSlider.value = 50;
      updateScoreUI(50);
    }
    if (mathInput) mathInput.value = '';
    
    document.querySelectorAll('.field-error').forEach(e => e.classList.remove('visible'));
    if (predictorSection) {
      window.scrollTo({ top: predictorSection.offsetTop - 80, behavior: 'smooth' });
    }
  });
}

// ---- NAV LINKS SMOOTH SCROLL ----
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', (e) => {
    const id = link.getAttribute('href').slice(1);
    const target = document.getElementById(id);
    if (!target) return;
    e.preventDefault();
    if (navLinks) navLinks.classList.remove('open');
    window.scrollTo({ top: target.offsetTop - 68, behavior: 'smooth' });
  });
});

// ---- UTILITIES ----
function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : '';
}
function titleCase(str) {
  return str ? str.split(' ').map(w => capitalize(w)).join(' ') : '';
}
function animateNumber(el, from, to, duration, formatter = v => v) {
  const start = performance.now();
  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = formatter(Math.round(from + (to - from) * eased));
    if (progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ---- SPINNER KEYFRAME (injected inline for btn animation) ----
const style = document.createElement('style');
style.textContent = `@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`;
document.head.appendChild(style);
