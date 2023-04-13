curve2 <- function(func, from, to, col = "black", lty = 1, lwd = 1, add = FALSE, xlab = NULL, ylab = NULL, xlim = NULL, ylim = NULL, main = NULL, ...) {
	f <- function(x) func(x)
	curve(expr = f, from = from, to = to, col = col, lty = lty, lwd = lwd, add = add, xlab = xlab, ylab = ylab, xlim = xlim, ylim = ylim, main = main, ... = ...)
}



loadResultsOrCompute <- function(file, computeExpr) {
	use_rds <- grepl(pattern = "rds$", x = file, ignore.case = TRUE)
	
	fn_save <- function(obj, file) {
		if (use_rds) {
			base::saveRDS(object = obj, file = file)
		} else {
			write.table(x = obj, file = file, quote = TRUE, sep = ";", dec = ".", row.names = FALSE, col.names = TRUE,  fileEncoding = "UTF-8")
		}
		obj
	}
	
	fn_read <- function(file) {
		if (use_rds) {
			base::readRDS(file = file)
		} else {
			read.table(file = file, header = TRUE, sep = ";", dec = ".", fileEncoding = "UTF-8", encoding = "UTF-8")
		}
	}
	
	
	file <- base::normalizePath(file, mustWork = FALSE)
	if (file.exists(file)) {
		return(fn_read(file = file))
	}
	
	res <- base::tryCatch(
		expr = computeExpr, error = function(cond) cond)
	
	# 'res' may have more than one class.
	if (any(class(res) %in% c("simpleError", "error", "condition"))) {
		print(traceback())
		stop(paste0("The computation failed: ", res))
	}
	
	fn_save(obj = res, file = file)
}


doWithParallelClusterExplicit <- function(cl, expr, errorValue = NULL, stopCl = TRUE) {
	doSNOW::registerDoSNOW(cl = cl)
	mev <- missing(errorValue)
	
	tryCatch(expr, error = function(cond) {
		if (!mev) {
			return(errorValue)
		}
		return(cond)
	}, finally = {
		if (stopCl) {
			parallel::stopCluster(cl)
			foreach::registerDoSEQ()
			gc()
		}
	})
}


doWithParallelCluster <- function(expr, errorValue = NULL, numCores = parallel::detectCores()) {
	cl <- parallel::makePSOCKcluster(numCores)
	doWithParallelClusterExplicit(cl = cl, expr = expr, errorValue = errorValue, stopCl = TRUE)
}
