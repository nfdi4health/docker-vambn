# originally from plot_all_marginals_paper.R
# This is the analysis file for the decoded VP data.

rm(list = ls())
library(tidyverse)
library(plyr)
library(grid)
library(gridExtra)
library(Matching)
library(LaplacesDemon)
library(Hmisc)
library(corrplot)
library(Hmisc)
library(ggpubr)

########## Name output files

config <- config::get(file = "/vambn/02_config/config_r.yml")
path_data_in <- config$path_data_in

name <- config$name
data_out <- paste0(config$path_data_out, name)
data_all <- readRDS(paste0(config$path_data_in, "data_condensed.rds"))
virtual <- readRDS(paste0(data_out, "_VirtualPPts.rds"))
real <- readRDS(paste0(data_out, "_RealPPts.rds"))
dec_rp <- read.csv(paste0(config$path_hivae, "reconRP.csv"))
dec_vp <- read.csv(paste0(config$path_hivae, "decodedVP.csv"))
folder_a <- paste0(config$path_data_out, "Plots/A/")
folder_b <- paste0(config$path_data_out, "Plots/B/")
folder_c <- paste0(config$path_data_out, "Plots/C/")


create_all_plots <- function() {
    ############ Plot A: Distribution of Encoded Real vs Encoded Virtual

    orig <- data_all[!grepl("stalone", names(data_all))]
    orig <- orig %>% reduce(merge, by="SUBJID")
    colnames(orig) <- gsub(" ", "\\.", colnames(orig))
    colnames(orig) <- gsub("[()]", "\\.", colnames(orig))

    orig <- orig[, (colnames(orig) %in% colnames(dec_rp))]
    for (col in colnames(orig)) {
        if (is.factor(orig[, col])) {
            orig[, col] <- factor(orig[, col])
            lvs <- levels(orig[, col])
            names(lvs) <- as.character(0:(length(lvs) - 1))
            dec_rp[,col] <- factor(dec_rp[, col], labels = unlist(lvs[levels(factor(dec_rp[, col]))]))
            dec_vp[,col] <- factor(dec_vp[, col], labels = unlist(lvs[levels(factor(dec_vp[,col]))]))
        }
    }

    orig$SUBJID <- NULL
    real$SUBJID <- NULL
    dec_rp$SUBJID <- NULL
    dec_vp$SUBJID <- NULL

    orig$type <- "real"
    virtual$type <- "virtual"
    real$type <- "real"
    dec_vp$type <- "decoded virtual"
    dec_rp$type <- "decoded real"

    # encoded real vs encoded virtual
    codes <- rbind(virtual, real)
    if (!all(sort(colnames(virtual)) == sort(colnames(real))))
        error("Unmatched column names!")

    codes$type <- factor(codes$type, levels = c("real", "virtual"))
    compare_dists(codes, "type", folder_a, 0.95)

    ############ Plot B: Distribution of Real vs Decoded Real vs Decoded Virtual

    orig <- orig[, sort(colnames(orig))]
    dec_rp <- dec_rp[, sort(colnames(dec_rp))]
    dec_vp <- dec_vp[, sort(colnames(dec_vp))]
    all <- rbind(orig, dec_rp, dec_vp)
    all$type <- factor(all$type, levels = c("real", "decoded real", "decoded virtual"))
    compare_dists_paper(all, "type", folder_b)

    ############ Plot C: Correlation of Original and Decoded Real and Decoded Virtual

    rm(list = ls())
    orig <- list(data_all[["snp_VIS1"]],
               data_all[["csf_VIS1"]],
               data_all[["volume_VIS1"]],
               data_all[["volume_VIS6"]],
               data_all[["volume_VIS12"]],
               data_all[["volume_VIS24"]],
               data_all[["cogtest_VIS1"]],
               data_all[["cogtest_VIS6"]],
               data_all[["cogtest_VIS12"]],
               data_all[["cogtest_VIS24"]],
               data_all[["brain68_VIS1"]]
    )

    orig <- orig %>% reduce(merge, by = "SUBJID")
    colnames(orig) <- gsub(" |,|-", "\\.", colnames(orig))
    colnames(orig) <- gsub("[()]", "\\.", colnames(orig))
    orig <- orig[, (colnames(orig) %in% colnames(dec_rp))]

    for (col in colnames(orig)) {
        if (is.factor(orig[, col]) |grepl("COGT_RAVLT.learning|COGT_RAVLT.forgetting|COGT_FAQ|COGT_MMSE", col)) {
            orig[, col] <- factor(orig[, col])
            lvs <- levels(orig[, col])
            names(lvs) <- as.character(0:(length(lvs) - 1))
            dec_rp[, col] <- factor(dec_rp[, col], labels = unlist(lvs[levels(factor(dec_rp[, col]))]))
            dec_vp[, col] <- factor(dec_vp[, col], labels = unlist(lvs[levels(factor(dec_vp[, col]))]))
        }
    }

    orig <- orig[, colnames(dec_rp)]
    orig <- orig[, sapply(colnames(orig), function(x) !is.factor(orig[, x]))]
    dec_rp <- dec_rp[, sapply(colnames(dec_rp), function(x) !is.factor(dec_rp[, x]))]
    dec_vp <- dec_vp[, sapply(colnames(dec_vp), function(x) !is.factor(dec_vp[, x]))]

    if (!((all((colnames(dec_rp)) == (colnames(orig)))) & (all((colnames(dec_rp)) == (colnames(dec_vp)))) & (all((colnames(orig)) == (colnames(dec_vp))))))
        warning("columns not matching!")

    oc <- rcorr(as.matrix(orig))
    rc <- rcorr(as.matrix(dec_rp))
    vc <- rcorr(as.matrix(dec_vp))

    norm(oc$r, type = "F")
    norm(rc$r, type = "F")
    norm(vc$r, type = "F")
    norm(oc$r - vc$r, type = "F") / norm(oc$r, type = "F")
    norm(oc$r - rc$r, type = "F") / norm(oc$r, type = "F")

    pdf(paste0(folder_c, "orig.pdf"))
    corrplot(oc$r, type = "upper", tl.cex = 0.25, title = paste0("Original (Norm: ", round(norm(oc$r, type = "F"), 2), ")"), mar = c(0,0,3,0))
    dev.off()

    pdf(paste0(folder_c, "dec_rp.pdf"))
    corrplot(rc$r, type = "upper", tl.cex = 0.25, title = paste0("Decoded real (Norm: ", round(norm(rc$r, type = "F"), 2), ")"), mar = c(0,0,3,0))
    dev.off()

    pdf(paste0(folder_c, "dec_vp.pdf"))
    corrplot(vc$r, type = "upper", tl.cex = 0.25, title = paste0("Decoded virtual (Norm: ", round(norm(vc$r, type = "F"), 2), ")"), mar = c(0,0,3,0))
    dev.off()
}


# originally from helper/compare_dists.R
compare_dists <- function(data, typecol, folder, conf.lev) {
  vnames <- colnames(data[, !grepl("type", colnames(data))])
  alpha <- 0.05 / length(vnames)

  lv <- levels(factor(data[, typecol]))
  group.1 <- subset(data, eval(parse(text = typecol)) == lv[1])
  group.2 <- subset(data, eval(parse(text = typecol)) == lv[2])

  pvals <- NULL
  for (col in vnames) {
    # ks if cont, chisq if cat
    pval <- tryCatch({
      ifelse(
        is.factor(group.1[, col]),
        chisq.test(group.1[, col], group.2[, col], simulate.p.value = TRUE)$p.value, ks.boot(group.1[, col], group.2[, col])$ks.boot.pvalue
      )
    }, error = function(e) {
      print(e)
      NA # for when no instance of a level in factor
    })

    dat <- data[, grepl(paste0(col, "|type"), colnames(data))]
    sig <- ifelse(pval < alpha, "*", "N.S.")

    if (is.numeric(data[, col])) {
      dv <- group.1[, col]
      if ((group.1$type[1] != "real") & (group.1$type[1] != "dec_rp"))
        error("Check variable levels!")
      z <- qnorm((1 - conf.lev) / 2, lower.tail = F)
      conf <- c(mean(dv, na.rm = T) - z * sd(dv, na.rm = T), mean(dv, na.rm = T) + z * sd(dv, na.rm = T))
      plot <- ggplot(dat, aes(x = eval(parse(text = col)), fill = type)) + geom_density(alpha = .2) + xlab(col) + ylab("density") + ggtitle(paste("Permutation Pval:", pval,"Alpha:", signif(alpha, digits = 3), "Sig:", sig)) + geom_vline(xintercept = c(conf[1], mean(dv,na.rm = T), conf[2]), linetype = "dashed")
    }else {
      gd <- dat %>% group_by(type) %>% count
      plot <- ggplot(gd, aes(x = eval(parse(text = col)), y = freq, fill = type)) + geom_bar(position = "dodge", stat = "identity") + scale_fill_brewer(palette = "Dark2") + xlab(col) + ylab("Count") + ggtitle(paste("Permutation Pval:", pval, "Alpha:", signif(alpha, digits = 3), "Sig:", sig))
    }

    pvals <- c(pvals, pval)
    ggsave(paste0(folder, col, ".png"), plot, device = "png")
  }

  hist(pvals, breaks = 250)
  abline(v = alpha, col = "red")
  abline(v = 0.05, col = "blue")
  print(paste("Corrected: Significant difference in", length(pvals[pvals<alpha]), "out of", length(pvals)))
  print(paste("Uncorrected: Significant difference in", length(pvals[pvals<0.05]), "out of", length(pvals)))
}


# originally from helper/compare_dists.R
compare_dists_paper <- function(data, typecol, folder) {
  vnames <- colnames(data[, !grepl("type", colnames(data))])
  typecol <- "type"
  lv <- levels(factor(data[, typecol]))
  group.1 <- subset(data, eval(parse(text = typecol)) == lv[1])
  group.2 <- subset(data, eval(parse(text = typecol)) == lv[2])
  group.3 <- subset(data, eval(parse(text = typecol)) == lv[3])

  print(paste0("1: ", lv[1], " 2: ", lv[2], " 3: ", lv[3]))

  pvals <- NULL
  for (col in vnames) {
      print(col)
      type <- "type"
      dat <- data[, grepl(paste0(col, "$|type"), colnames(data))]

      if (is.numeric(data[, col])) {
          dv1 <- group.1[, col]
          dv2 <- group.2[, col]
          dv3 <- group.3[, col]
          set.seed(123 + grep(col, colnames(data)))
          px <- dnorm(runif(dv2))
          py <- dnorm(runif(dv3))
          kld <- KLD(px, py)

          klValue <- round(kld$sum.KLD.py.px, 4)
          stats <- matrix(c(lv[1], round(mean(dv1, na.rm = T), 2), round(sd(dv1, na.rm = T), 2), round(quantile(dv1, na.rm = T)["25%"], 2), round(median(dv1, na.rm = T), 2), round(quantile(dv1, na.rm = T)["75%"], 2),
                      lv[2], round(mean(dv2, na.rm = T), 2), round(sd(dv2, na.rm = T), 2), round(quantile(dv2, na.rm = T)["25%"], 2), round(median(dv2, na.rm = T), 2), round(quantile(dv2, na.rm = T)["75%"], 2),
                      lv[3], round(mean(dv3, na.rm = T), 2), round(sd(dv3, na.rm = T), 2), round(quantile(dv3, na.rm = T)["25%"], 2), round(median(dv3, na.rm = T), 2), round(quantile(dv3, na.rm = T)["75%"], 2)), ncol = 6, byrow = TRUE)
          colnames(stats) <- c("Type", "Mean", "SD", "25%", "Median", "75%")
          plot1 <- ggplot(dat, aes(x = type, y = eval(parse(text = col)), fill = type)) + geom_violin(position = position_dodge(1)) + geom_boxplot(position = position_dodge(1), width=0.05) + ggtitle(paste("KL divergence:", klValue)) + xlab("type") + ylab(col) + theme(legend.position = "None", axis.title.x = element_blank())
          gr <- grey.colors(3)
          tt3 <- ttheme_minimal(core = list(bg_params = list(fill = c(gr[2], gr[3], gr[2]), col = NA), fg_params = list(fontface = 3)))
          plot <- grid.arrange(plot1, tableGrob(stats, theme = tt3), ncol = 1, widths = unit(15, "cm"), heights = unit(c(8, 3), c("cm", "cm")))
      }else{
          gd <- dat %>% group_by(type) %>% count
          plot <- ggplot(gd, aes(x = eval(parse(text = col)), y  =freq, fill = type)) + geom_bar(position = "dodge", stat = "identity") + scale_fill_brewer(palette = "Dark2") + xlab(col) + ylab("Count")
      }
    ggsave(paste0(folder, col, ".png"), plot, device = "png", width = 7.26, height = 4.35)
    ggsave(paste0(folder, col, ".eps"), plot, device = "eps", width = 7.26, height = 4.35)
  }
}