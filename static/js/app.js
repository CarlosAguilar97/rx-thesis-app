function formatearProbabilidad(valor) {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return "—";
  return Number(valor).toFixed(2);
}

function setPrediction(prefix, pred1Label, pred1Prob, pred2Label, pred2Prob) {
  const p1Label = document.getElementById(`${prefix}_pred1_label`);
  const p1Prob = document.getElementById(`${prefix}_pred1_prob`);
  const p2Label = document.getElementById(`${prefix}_pred2_label`);
  const p2Prob = document.getElementById(`${prefix}_pred2_prob`);
  const prog1 = document.getElementById(`${prefix}_progress1`);
  const prog2 = document.getElementById(`${prefix}_progress2`);

  p1Label.textContent = pred1Label || "—";
  p1Prob.textContent = formatearProbabilidad(pred1Prob);

  p2Label.textContent = pred2Label || "—";
  p2Prob.textContent = formatearProbabilidad(pred2Prob);

  prog1.value = pred1Prob ? Math.round(Number(pred1Prob) * 100) : 0;
  prog2.value = pred2Prob ? Math.round(Number(pred2Prob) * 100) : 0;
}

function renderGradcams(gradcamImages) {
  const section = document.getElementById("gradcamSection");
  const grid = document.getElementById("gradcamGrid");
  grid.innerHTML = "";

  if (!gradcamImages || Object.keys(gradcamImages).length === 0) {
    section.classList.add("hidden");
    return;
  }

  Object.entries(gradcamImages).forEach(([label, src]) => {
    const item = document.createElement("div");
    item.className = "rounded-2xl bg-base-100 border border-base-300 p-3";

    item.innerHTML = `
      <p class="font-semibold mb-2">${label}</p>
      <img src="${src}" alt="GradCAM ${label}" class="rounded-xl w-full object-contain" />
    `;
    grid.appendChild(item);
  });

  section.classList.remove("hidden");
}

function limpiarVista() {
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("previewImage");
  const placeholder = document.getElementById("placeholderText");
  const estado = document.getElementById("estadoSistema");
  const originalDescription = document.getElementById("originalDescription").value;

 

  fileInput.value = "";
  originalDescription.value = "";
  preview.src = "";
  preview.classList.add("hidden");
  placeholder.classList.remove("hidden");

  document.getElementById("calidadValor").textContent = "—";
  document.getElementById("inferenciaValor").textContent = "—";
  document.getElementById("originalDescription").textContent = "—";
  setPrediction("patterns", "—", null, "—", null);
  setPrediction("diseases", "—", null, "—", null);
  renderGradcams(null);

  estado.textContent = "Estado: esperando imagen";
  estado.className = "badge badge-ghost";
}

async function analizarImagen() {
  const fileInput = document.getElementById("fileInput");
  const estado = document.getElementById("estadoSistema");
  const preview = document.getElementById("previewImage");
  const placeholder = document.getElementById("placeholderText");
  const inferenciaValor = document.getElementById("inferenciaValor");
  const calidadValor = document.getElementById("calidadValor");
  const originalDescription = document.getElementById("originalDescription").value;

  if (!fileInput.files.length) {
    alert("Selecciona una imagen");
    return;
  }

  if (originalDescription.length === 0) {
    alert("Debe escribir la descripción original de la radiografía.");
    return;
  }


  const file = fileInput.files[0];

  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  placeholder.classList.add("hidden");

  calidadValor.textContent = `${file.type || "imagen"} • ${(file.size / 1024 / 1024).toFixed(2)} MB`;

  estado.textContent = "Estado: procesando...";
  estado.className = "badge badge-warning";
  inferenciaValor.textContent = "Procesando...";

  setPrediction("patterns", "Procesando...", null, "Procesando...", null);
  setPrediction("diseases", "Procesando...", null, "Procesando...", null);
  renderGradcams(null);

  const startTime = performance.now();


  const formData = new FormData();
  formData.append("file", file);
  formData.append("original_description", originalDescription);


  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "No se pudo completar el análisis");
    }

    const data = await response.json();
    const endTime = performance.now();
    const seconds = ((endTime - startTime) / 1000).toFixed(2);

    setPrediction(
      "patterns",
      data.patterns?.pred1_label,
      data.patterns?.pred1_prob,
      data.patterns?.pred2_label,
      data.patterns?.pred2_prob
    );

    setPrediction(
      "diseases",
      data.diseases?.pred1_label,
      data.diseases?.pred1_prob,
      data.diseases?.pred2_label,
      data.diseases?.pred2_prob
    );

    renderGradcams(data.patterns?.gradcam_images);

    inferenciaValor.textContent = `${seconds}s`;
    estado.textContent = "Estado: análisis completado";
    estado.className = "badge badge-success";

  } catch (error) {
    console.error(error);
    inferenciaValor.textContent = "Error";
    estado.textContent = "Estado: error en análisis";
    estado.className = "badge badge-error";

    setPrediction("patterns", "Error", null, "Error", null);
    setPrediction("diseases", "Error", null, "Error", null);
    renderGradcams(null);

    alert(error.message || "Ocurrió un error al analizar la imagen");
  }
}