# ============================================================
# visualisations_ggplot.R
# Projet : Analyse des déterminants de la satisfaction
#          en maison de retraite
# Description : Visualisations publication-ready avec ggplot2
# ============================================================


# ─────────────────────────────────────────────
# 0. PACKAGES
# ─────────────────────────────────────────────

if (!require(ggplot2))  install.packages("ggplot2")
if (!require(dplyr))    install.packages("dplyr")
if (!require(tidyr))    install.packages("tidyr")
if (!require(readxl))   install.packages("readxl")

library(ggplot2)
library(dplyr)
library(tidyr)
library(readxl)

# Créer le dossier de figures si nécessaire
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)

# Thème personnalisé
theme_retraite <- theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle   = element_text(hjust = 0.5, color = "grey50"),
    axis.title      = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )


# ─────────────────────────────────────────────
# CHARGEMENT & NETTOYAGE
# ─────────────────────────────────────────────

df <- read_excel("data/raw/Retraite.xlsx")
names(df) <- c("Formule", "Public", "Activite", "Accueil", "Soins_Qualite",
               "Competence_Personnel", "Disponibilite_Personnel", "Reconfort",
               "Qualite_Restauration", "Hygiene", "Confort_Chambre",
               "Services_Annexes", "Satisfaction", "PRIX_PENSION", "Sexe", "CSP")

df$Sexe   <- factor(df$Sexe, levels = c("Un homme", "Une femme"),
                    labels = c("homme", "femme"))
df$Formule <- as.factor(df$Formule)
df$Public  <- as.factor(df$Public)

# Imputation médiane
imputer_mediane <- function(x) { x[is.na(x)] <- median(x, na.rm=TRUE); x }
cols_num <- c("Accueil","Soins_Qualite","Competence_Personnel",
              "Disponibilite_Personnel","Reconfort","Qualite_Restauration",
              "Hygiene","Confort_Chambre","Services_Annexes")
df[cols_num] <- lapply(df[cols_num], imputer_mediane)

cat("✅ Données prêtes pour les visualisations\n")


# ─────────────────────────────────────────────
# GRAPHIQUE 1 : Distribution de la Satisfaction
# ─────────────────────────────────────────────

p1 <- ggplot(df, aes(x = Satisfaction)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white", alpha = 0.85) +
  geom_vline(aes(xintercept = mean(Satisfaction)),
             color = "red", linetype = "dashed", linewidth = 1,
             show.legend = TRUE) +
  geom_vline(aes(xintercept = median(Satisfaction)),
             color = "orange", linetype = "dashed", linewidth = 1) +
  annotate("text", x = mean(df$Satisfaction) + 0.3,
           y = 40, label = paste("Moy =", round(mean(df$Satisfaction), 1)),
           color = "red", hjust = 0, size = 3.5) +
  annotate("text", x = median(df$Satisfaction) - 0.3,
           y = 35, label = paste("Méd =", round(median(df$Satisfaction), 1)),
           color = "orange", hjust = 1, size = 3.5) +
  labs(title    = "Distribution de la Satisfaction globale",
       subtitle = "300 résidents — note sur 10",
       x = "Note de satisfaction",
       y = "Nombre de résidents") +
  theme_retraite

ggsave("reports/figures/gg_satisfaction_globale.png", p1, width = 8, height = 5, dpi = 150)
print(p1)
cat("✅ Graphique 1 sauvegardé\n")


# ─────────────────────────────────────────────
# GRAPHIQUE 2 : Boxplot — Satisfaction par Public
# ─────────────────────────────────────────────

p2 <- ggplot(df, aes(x = Public, y = Satisfaction, fill = Public)) +
  geom_boxplot(alpha = 0.8, outlier.color = "red", outlier.size = 2) +
  geom_jitter(width = 0.15, alpha = 0.2, size = 1.2) +
  scale_fill_manual(values = c("lightgreen", "orange", "tomato")) +
  labs(title    = "Satisfaction selon le degré d'autonomie",
       subtitle = "ANOVA significative (p < 0.01) — Valides moins satisfaits",
       x = "Type de Public",
       y = "Satisfaction (sur 10)") +
  theme_retraite +
  theme(legend.position = "none")

ggsave("reports/figures/gg_boxplot_public.png", p2, width = 8, height = 5, dpi = 150)
print(p2)
cat("✅ Graphique 2 sauvegardé\n")


# ─────────────────────────────────────────────
# GRAPHIQUE 3 : Bar chart — Satisfaction par service
# ─────────────────────────────────────────────

services_moy <- df %>%
  select(all_of(cols_num)) %>%
  summarise(across(everything(), mean)) %>%
  pivot_longer(everything(), names_to = "Service", values_to = "Moyenne") %>%
  mutate(
    Service  = gsub("_", " ", Service),
    Couleur  = ifelse(Moyenne < mean(Moyenne), "Sous la moyenne", "Au-dessus")
  ) %>%
  arrange(Moyenne)

services_moy$Service <- factor(services_moy$Service, levels = services_moy$Service)

p3 <- ggplot(services_moy, aes(x = Moyenne, y = Service, fill = Couleur)) +
  geom_col(alpha = 0.85) +
  geom_vline(xintercept = mean(services_moy$Moyenne),
             linetype = "dashed", color = "black", linewidth = 1) +
  geom_text(aes(label = round(Moyenne, 2)), hjust = -0.2, size = 3.5) +
  scale_fill_manual(values = c("Au-dessus" = "#2980b9", "Sous la moyenne" = "#e74c3c")) +
  labs(title    = "Satisfaction moyenne par service (sur 5)",
       subtitle = "Rouge = sous la moyenne | Bleu = au-dessus",
       x = "Note moyenne",
       y = "",
       fill = "") +
  xlim(0, 5.5) +
  theme_retraite

ggsave("reports/figures/gg_satisfaction_services.png", p3, width = 10, height = 6, dpi = 150)
print(p3)
cat("✅ Graphique 3 sauvegardé\n")


# ─────────────────────────────────────────────
# GRAPHIQUE 4 : Scatter — Prix vs Satisfaction
# ─────────────────────────────────────────────

p4 <- ggplot(df, aes(x = PRIX_PENSION, y = Satisfaction)) +
  geom_point(alpha = 0.35, color = "darkgreen", size = 2) +
  geom_smooth(method = "lm", color = "red", linewidth = 1.2, se = TRUE) +
  annotate("text", x = max(df$PRIX_PENSION) * 0.6, y = 9.5,
           label = paste("r =", round(cor(df$PRIX_PENSION, df$Satisfaction), 3)),
           size = 4.5, color = "red", fontface = "bold") +
  labs(title    = "Relation entre Prix de la pension et Satisfaction",
       subtitle = "Droite quasi horizontale : le prix n'influence pas la satisfaction",
       x = "Prix de la pension",
       y = "Satisfaction (sur 10)") +
  theme_retraite

ggsave("reports/figures/gg_prix_satisfaction.png", p4, width = 8, height = 5, dpi = 150)
print(p4)
cat("✅ Graphique 4 sauvegardé\n")


# ─────────────────────────────────────────────
# GRAPHIQUE 5 : Prédictions (Scénarios)
# ─────────────────────────────────────────────

modele_final <- lm(Satisfaction ~ Soins_Qualite + Competence_Personnel +
                     Reconfort + Hygiene, data = df)

nouveaux <- data.frame(
  Soins_Qualite = c(3, 5), Competence_Personnel = c(3, 5),
  Reconfort = c(3, 5),     Hygiene = c(3, 5)
)

pred <- predict(modele_final, newdata = nouveaux, interval = "confidence")
pred_df <- data.frame(
  Profil = c("Profil Moyen\n(3/5 partout)", "Profil Excellent\n(5/5 partout)"),
  fit    = pred[, "fit"],
  lwr    = pred[, "lwr"],
  upr    = pred[, "upr"]
)

p5 <- ggplot(pred_df, aes(x = Profil, y = fit, fill = Profil)) +
  geom_col(width = 0.5, alpha = 0.85) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), width = 0.15, linewidth = 1) +
  geom_text(aes(label = paste0(round(fit, 2), "/10")),
            vjust = -0.8, fontface = "bold", size = 5) +
  scale_fill_manual(values = c("#f39c12", "#27ae60")) +
  ylim(0, 11) +
  labs(title    = "Satisfaction prédite par scénario",
       subtitle = "Intervalles de confiance à 95%",
       x = "",
       y = "Note de satisfaction prédite (sur 10)") +
  theme_retraite +
  theme(legend.position = "none")

ggsave("reports/figures/gg_predictions.png", p5, width = 7, height = 5, dpi = 150)
print(p5)
cat("✅ Graphique 5 sauvegardé\n")


# ─────────────────────────────────────────────
# GRAPHIQUE 6 : Satisfaction par Sexe et Public
# ─────────────────────────────────────────────

p6 <- ggplot(df, aes(x = Public, y = Satisfaction, fill = Sexe)) +
  geom_boxplot(alpha = 0.75, outlier.alpha = 0.3) +
  scale_fill_manual(values = c("homme" = "#3498db", "femme" = "#e91e8c")) +
  labs(title    = "Satisfaction par Public et par Sexe",
       subtitle = "Analyse croisée autonomie × genre",
       x = "Type de Public",
       y = "Satisfaction (sur 10)",
       fill = "Sexe") +
  theme_retraite

ggsave("reports/figures/gg_public_sexe.png", p6, width = 9, height = 5, dpi = 150)
print(p6)
cat("✅ Graphique 6 sauvegardé\n")

cat("\n✅ Toutes les visualisations ggplot2 sont sauvegardées dans reports/figures/\n")
