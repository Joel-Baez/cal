const form = document.getElementById('integralForm');
const mathfieldHost = document.getElementById('mathField');
const mathInput = document.getElementById('mathInput');
const latexPreview = document.getElementById('latexPreview');
const variableInput = document.getElementById('variableInput');
const variableDisplay = document.getElementById('variableDisplay');
const statusMessage = document.getElementById('statusMessage');
const analysisCard = document.getElementById('analysisCard');
const methodSuggestion = document.getElementById('methodSuggestion');
const exampleCard = document.getElementById('exampleCard');
const keyboardTabs = document.getElementById('keyboardTabs');
const keyboardGrid = document.getElementById('keyboardGrid');
const examplePills = document.querySelectorAll('.pill');

const PLACEHOLDER_TOKEN = '\\placeholder{}';
const PLACEHOLDER_RENDER = '\\square';
const P = PLACEHOLDER_TOKEN;

const KEYBOARD_GROUPS = [
  {
    id: 'basico',
    label: 'Básico',
    keys: [
      { label: '7', latex: '7' }, { label: '8', latex: '8' }, { label: '9', latex: '9' },
      { label: '4', latex: '4' }, { label: '5', latex: '5' }, { label: '6', latex: '6' },
      { label: '1', latex: '1' }, { label: '2', latex: '2' }, { label: '3', latex: '3' },
      { label: '0', latex: '0' }, { label: '.', latex: '.' }, { label: ',', latex: ',' },
      { label: 'x', latex: 'x' }, { label: 'y', latex: 'y' }, { label: 't', latex: 't' },
      { label: 'u', latex: 'u' }, { label: 'θ', latex: '\\theta' }, { label: 'π', latex: '\\pi' },
      { label: 'e', latex: 'e' }, { label: '+', latex: '+' }, { label: '−', latex: '-' },
      { label: '·', latex: '\\cdot' }, { label: '÷', latex: '\\div' }, { label: '=', latex: '=' },
      { label: '(', latex: '(' }, { label: ')', latex: ')' }
    ]
  },
  {
    id: 'estructuras',
    label: 'Plantillas',
    keys: [
      { label: '\\frac{□}{□}', latex: `\\frac{${P}}{${P}}` },
      { label: '\\sqrt{}', latex: `\\sqrt{${P}}` },
      { label: '\\sqrt[□]{}', latex: `\\sqrt[${P}]{${P}}` },
      { label: '^', latex: `^{${P}}` },
      { label: '( )^{□}', latex: `\\left(${P}\\right)^{${P}}` },
      { label: 'e^{□}', latex: `e^{${P}}` },
      { label: 'e^{ax}', latex: `e^{${P} x}` },
      { label: '\\ln| |', latex: `\\ln\\left|${P}\\right|` },
      { label: '\\int f', latex: `\\int ${P} \\,d${P}` }
    ]
  },
  {
    id: 'trig',
    label: 'Trigonometría',
    keys: [
      { label: 'sin', latex: `\\sin\\left(${P}\\right)` },
      { label: 'cos', latex: `\\cos\\left(${P}\\right)` },
      { label: 'tan', latex: `\\tan\\left(${P}\\right)` },
      { label: 'sec', latex: `\\sec\\left(${P}\\right)` },
      { label: 'csc', latex: `\\csc\\left(${P}\\right)` },
      { label: 'cot', latex: `\\cot\\left(${P}\\right)` },
      { label: 'sinh', latex: `\\sinh\\left(${P}\\right)` },
      { label: 'cosh', latex: `\\cosh\\left(${P}\\right)` },
      { label: 'tanh', latex: `\\tanh\\left(${P}\\right)` },
      { label: 'arcsin', latex: `\\arcsin\\left(${P}\\right)` },
      { label: 'arccos', latex: `\\arccos\\left(${P}\\right)` },
      { label: 'arctan', latex: `\\arctan\\left(${P}\\right)` }
    ]
  },
  {
    id: 'estructuras-trig',
    label: 'Identidades',
    keys: [
      { label: '√(a²−x²)', latex: `\\sqrt{${P}^{2}-x^{2}}` },
      { label: '√(a²+x²)', latex: `\\sqrt{${P}^{2}+x^{2}}` },
      { label: '√(x²−a²)', latex: `\\sqrt{x^{2}-${P}^{2}}` },
      { label: 'sin²+cos²', latex: `\\sin^{2}(${P})+\\cos^{2}(${P})` },
      { label: '1+tan²', latex: `1+\\tan^{2}(${P})` },
      { label: 'sec²', latex: `\\sec^{2}(${P})` }
    ]
  },
  {
    id: 'fracciones',
    label: 'Fracciones parciales',
    keys: [
      { label: 'A/(x-a)', latex: `\\frac{A}{x-${P}}` },
      { label: 'B/(x+a)', latex: `\\frac{B}{x+${P}}` },
      { label: '(Cx+D)/(x²+a²)', latex: `\\frac{Cx+D}{x^{2}+${P}^{2}}` },
      { label: '(Ax+B)/(x-a)²', latex: `\\frac{Ax+B}{(x-${P})^{2}}` },
      { label: '1/(x(x+a))', latex: `\\frac{1}{x\\left(x+${P}\\right)}` },
      { label: 'P(x)/Q(x)', latex: `\\frac{${P}}{${P}}` }
    ]
  },
  {
    id: 'metodos',
    label: 'Métodos',
    keys: [
      { label: 'u(x)=', latex: `u(x)=${P}` },
      { label: 'du=', latex: `du=${P}\\,d${P}` },
      { label: 'v(x)=', latex: `v(x)=${P}` },
      { label: 'dv=', latex: `dv=${P}\\,d${P}` },
      { label: '∫u dv', latex: '\\int u\\,dv' },
      { label: '∫f(u)du', latex: '\\int f(u)\\,du' },
      { label: 'u→x', latex: `x=${P}(u)` },
      { label: 'θ→x', latex: `x=${P}(\\theta)` },
      { label: 'dx=', latex: `d${P}=${P}` }
    ]
  }
];

const METHOD_TO_GROUP = {
  substitution: 'metodos',
  parts: 'metodos',
  trig: 'trig',
  partial_fractions: 'fracciones',
  repeated_factors: 'fracciones'
};

let activeGroup = KEYBOARD_GROUPS[0].id;

const setStatus = (message, variant = 'idle') => {
  if (!statusMessage) return;
  statusMessage.textContent = message;
  statusMessage.className = `status ${variant}`;
};

const retypeset = () => {
  if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
    MathJax.typesetPromise();
  }
};

const sanitizeVariable = (value) => {
  const letters = (value || '').replace(/[^a-zA-Z]/g, '');
  return letters || 'x';
};

const updateVariableDisplay = () => {
  const sanitized = sanitizeVariable(variableInput.value);
  variableInput.value = sanitized;
  variableDisplay.textContent = sanitized;
  updatePreview();
};

const getLatexValue = () => (mathInput ? mathInput.value : '');

const setLatexValue = (value) => {
  if (!mathInput) {
    setStatus('No se pudo actualizar el integrando porque el editor no está disponible.', 'error');
    return;
  }
  mathInput.value = value || '';
  handleLatexInput();
  mathInput.focus();
};

const updatePreview = () => {
  if (!latexPreview) return;
  const latex = getLatexValue();
  const variable = sanitizeVariable(variableInput.value);
  latexPreview.innerHTML = latex
    ? `$$\\int ${latex}\\,d${variable}$$`
    : '<span class="preview-placeholder">Vista previa en LaTeX</span>';
  retypeset();
};

const handleLatexInput = () => {
  mathfieldHost?.classList.remove('invalid');
  updatePreview();
  setStatus(getLatexValue()
    ? 'Expresión actualizada. Pulsa «Sugerir método».'
    : 'Escribe un integrando para comenzar.',
    'idle'
  );
};

const transformSnippet = (snippet) => {
  if (!snippet) return { text: '', caretOffset: null, caretLength: 0 };
  let caretOffset = null;
  let caretLength = 0;
  let placedPlaceholder = false;
  let result = '';
  let remaining = snippet;

  while (remaining.length) {
    const index = remaining.indexOf(PLACEHOLDER_TOKEN);
    if (index === -1) { result += remaining; break; }
    const before = remaining.slice(0, index);
    result += before;
    if (caretOffset === null) caretOffset = result.length;
    result += PLACEHOLDER_RENDER;
    caretLength = PLACEHOLDER_RENDER.length;
    placedPlaceholder = true;
    remaining = remaining.slice(index + PLACEHOLDER_TOKEN.length);
  }
  if (!placedPlaceholder) { caretOffset = null; caretLength = 0; }
  return { text: result, caretOffset, caretLength };
};

const insertLatex = (rawSnippet) => {
  if (!mathInput) {
    setStatus('No se pudo insertar el símbolo porque el editor no está disponible.', 'error');
    return;
  }
  const { text: snippet, caretOffset, caretLength } = transformSnippet(rawSnippet);
  const { selectionStart = mathInput.value.length, selectionEnd = mathInput.value.length } = mathInput;
  const baseValue = mathInput.value;
  const prefix = baseValue.slice(0, selectionStart);
  const suffix = baseValue.slice(selectionEnd);
  mathInput.value = `${prefix}${snippet}${suffix}`;

  const insertionPoint = selectionStart + (caretOffset ?? snippet.length);
  requestAnimationFrame(() => {
    mathInput.focus();
    const selectionEndPoint = caretOffset === null ? insertionPoint : insertionPoint + caretLength;
    mathInput.setSelectionRange(insertionPoint, selectionEndPoint);
  });

  handleLatexInput();
};

const initializeLatexEditor = () => {
  if (!mathInput) {
    setStatus('No se encontró el editor de integrales en la página.', 'error');
    return;
  }
  const placeholder = mathInput.dataset.placeholder || '';
  mathInput.placeholder = placeholder;
  mathInput.value = '';
  mathInput.addEventListener('input', handleLatexInput);
  mathInput.addEventListener('focus', () => mathfieldHost?.classList.remove('invalid'));
  mathInput.addEventListener('blur', () => mathfieldHost?.classList.remove('invalid'));
  handleLatexInput();
};

const renderKeyboardTabs = () => {
  keyboardTabs.innerHTML = KEYBOARD_GROUPS.map(({ id, label }) => `
    <button type="button" class="tab ${id === activeGroup ? 'active' : ''}" data-group="${id}" role="tab">
      ${label}
    </button>
  `).join('');
};

const renderKeyboardKeys = () => {
  const group = KEYBOARD_GROUPS.find(({ id }) => id === activeGroup) ?? KEYBOARD_GROUPS[0];
  keyboardGrid.innerHTML = group.keys.map(({ label, latex }) => `
    <button type="button" class="key" data-latex="${latex}">${label}</button>
  `).join('');
};

const refreshKeyboard = () => { renderKeyboardTabs(); renderKeyboardKeys(); };

keyboardTabs.addEventListener('click', (event) => {
  if (!(event.target instanceof HTMLButtonElement)) return;
  const { group } = event.target.dataset;
  if (!group || group === activeGroup) return;
  activeGroup = group;
  refreshKeyboard();
});

keyboardGrid.addEventListener('click', (event) => {
  if (!(event.target instanceof HTMLButtonElement)) return;
  const { latex } = event.target.dataset;
  if (!latex) return;
  insertLatex(latex);
});

initializeLatexEditor();
refreshKeyboard();

examplePills.forEach((pill) => {
  pill.addEventListener('click', () => {
    const latex = pill.dataset.latex;
    if (!latex) return;
    setLatexValue(latex);
    setStatus('Ejemplo cargado. Ajusta la expresión si lo necesitas.', 'idle');
  });
});

const renderAnalysis = (analysis) => {
  const {
    latex_integral: latexIntegral = '',
    variable = 'x',
    detected_features: features = [],
    warnings = []
  } = analysis;

  const featureList = features.length
    ? `<ul class="feature-list">${features.map((item) => `<li>${item}</li>`).join('')}</ul>`
    : '<p>No se detectaron patrones especiales adicionales.</p>';

  const warningsList = warnings.length
    ? `<div class="warnings"><h4>Advertencias</h4><ul>${warnings.map((item) => `<li>${item}</li>`).join('')}</ul></div>`
    : '';

  analysisCard.innerHTML = `
    <h3>Análisis del integrando</h3>
    <p class="integral-display">$$${latexIntegral || ''}$$</p>
    <div class="analysis-details">
      <div><span class="detail-label">Variable principal</span><span class="detail-value">${variable}</span></div>
    </div>
    ${featureList}
    ${warningsList}
  `;
};

const renderMethod = (method) => {
  if (!method) {
    methodSuggestion.innerHTML = `
      <h3>Método sugerido</h3>
      <p>No fue posible determinar un método predominante. Simplifica el integrando y vuelve a intentarlo.</p>
    `;
    exampleCard.innerHTML = `
      <h3>Ejemplo guiado</h3>
      <p>No se generó un ejemplo porque no hay método sugerido.</p>
    `;
    return;
  }

  const {
    title,
    badge,
    summary,
    example_integral: exampleIntegral,
    example_solution: exampleSolution,
    steps = [],
    setup = [],
    key
  } = method;

  const stepsList = steps.length
    ? `<ol class="step-list">${steps.map(({ title, description, equations = [] }) => `
          <li>
            <h4>${title}</h4>
            <p>${description}</p>
            ${equations.length
              ? `<div class="equation-group">${equations
                  .map((eq) => `<p class="equation">$$${eq}$$</p>`).join('')}</div>`
              : ''}
          </li>
        `).join('')}</ol>`
    : '';

  const setupList = setup.length
    ? `<div class="setup"><h4>Datos clave</h4><div class="setup-grid">${setup
        .map(({ label, value }) => `
          <div class="setup-item">
            <span class="setup-label">${label}</span>
            <span class="setup-value">$$${value}$$</span>
          </div>
        `).join('')}</div></div>`
    : '';

  methodSuggestion.innerHTML = `
    <div class="badge">${badge}</div>
    <h3>${title}</h3>
    <p>${summary}</p>
  `;

  exampleCard.innerHTML = `
    <h3>Ejemplo guiado</h3>
    <p class="similar">Integral modelo:</p>
    <p class="similar-example">$$${exampleIntegral}$$</p>
    <p class="similar">Resultado esperado:</p>
    <p class="similar-example">$$${exampleSolution}$$</p>
    ${setupList}
    ${stepsList}
    <p class="hint"><em>Nota:</em> todas las antiderivadas incluyen <strong>+ c</strong>.</p>
  `;

  highlightKeyboardGroup(key);
};

const highlightKeyboardGroup = (methodKey) => {
  const group = METHOD_TO_GROUP[methodKey] || 'basico';
  if (activeGroup !== group) activeGroup = group;
  refreshKeyboard();
};

const requestAnalysis = async (payload) => {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Error inesperado en el servidor.');
  }
  return response.json();
};

const updateVariableDisplayAndPreview = () => {
  updateVariableDisplay();
  updatePreview();
};

updateVariableDisplayAndPreview();
variableInput.addEventListener('input', updateVariableDisplay);

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const latexExpression = getLatexValue().trim();
  const variable = sanitizeVariable(variableInput.value);

  if (!latexExpression) {
    setStatus('Por favor completa el integrando en el editor antes de analizar.', 'error');
    mathfieldHost?.classList.add('invalid');
    return;
  }
  if (latexExpression.includes('\\placeholder') || latexExpression.includes('\\square')) {
    setStatus('Completa los espacios vacíos del teclado antes de enviar la integral.', 'error');
    mathfieldHost?.classList.add('invalid');
    return;
  }

  setStatus('Analizando la integral en el servidor de Python…', 'loading');

  try {
    const payload = { expression: latexExpression, variable };
    const data = await requestAnalysis(payload);
    if (data.status !== 'ok') throw new Error(data.error || 'No se pudo procesar la integral.');
    renderAnalysis(data.analysis);
    renderMethod(data.method);
    setStatus('Análisis completado. Revisa el método sugerido y el ejemplo.', 'success');
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Ocurrió un error al procesar la integral.', 'error');
    renderMethod(null);
  } finally {
    retypeset();
  }
});

retypeset();
