// Petit script : affiche la valeur des curseurs et remplit le formulaire avec un profil d'exemple.
(function () {
  const form = document.getElementById("formulaire");
  const champExemple = document.getElementById("exemple");

  // 1) Afficher la valeur à côté de chaque curseur (âge, heures)
  function majCurseur(curseur) {
    const sortie = document.getElementById(curseur.id + "-val");
    if (sortie) sortie.textContent = curseur.value;
  }
  form.querySelectorAll('input[type="range"]').forEach(function (c) {
    c.addEventListener("input", function () { majCurseur(c); });
    majCurseur(c);
  });

  // 2) Si la personne modifie le formulaire à la main, ce n'est plus un profil d'exemple
  form.addEventListener("input", function (e) {
    if (e.isTrusted) champExemple.value = "";
  });

  // 3) Clic sur un profil d'exemple : remplir tous les champs puis envoyer le formulaire
  document.querySelectorAll("[data-profil]").forEach(function (bouton) {
    bouton.addEventListener("click", function () {
      const profil = JSON.parse(bouton.dataset.profil);
      Object.keys(profil).forEach(function (nom) {
        const champ = form.elements[nom];
        if (!champ) return;
        if (champ.type === "checkbox") champ.checked = !!profil[nom];
        else champ.value = profil[nom];   // marche aussi pour les boutons radio (Homme / Femme)
      });
      form.querySelectorAll('input[type="range"]').forEach(majCurseur);
      champExemple.value = bouton.dataset.id;
      form.requestSubmit();
    });
  });

  // 4) Sur téléphone : après une prédiction, descendre automatiquement vers le résultat
  const resultat = document.querySelector("#resultat .resultat");
  if (resultat && window.matchMedia("(max-width: 980px)").matches) {
    resultat.scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();