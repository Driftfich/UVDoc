package com.example.uvdoc

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class ImageProcessor(context: Context) {

    private var interpreter: Interpreter? = null
    private val modelName = "model.tflite"

    // Model dims
    private val INPUT_SIZE = 256
    private val NUM_INPUT_CHANNELS = 5 // R, G, B, X, Y
    private val NUM_OUTPUT_CHANNELS = 6 // U, V, Mask, R_sh, G_sh, B_sh

    init {
        try {
            val assetManager = context.assets
            val modelDescriptor = assetManager.openFd(modelName)
            val inputStream = java.io.FileInputStream(modelDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = modelDescriptor.startOffset
            val declaredLength = modelDescriptor.declaredLength
            val modelBuffer = fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options()
            options.setUseXNNPACK(true)

            interpreter = Interpreter(modelBuffer, options)

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // Basic Linear Algebra Classes for TPS
    data class Point(val x: Float, val y: Float)

    class TPSSolver {
        private var weights: FloatArray? = null
        private var affine: FloatArray? = null // 3x2 -> 6 floats
        private var controlPoints: List<Point>? = null
        private var lambda = 0.01f

        fun fit(sources: List<Point>, targets: List<Point>) {
            val n = sources.size
            if (n == 0) return

            controlPoints = sources

            // 1. Build L Matrix (N+3 x N+3)
            // [ K  P ]
            // [ P^T 0 ]

            val matL = Array(n + 3) { FloatArray(n + 3) }

            // Fill K (N x N)
            for (i in 0 until n) {
                for (j in i until n) {
                    val dx = sources[i].x - sources[j].x
                    val dy = sources[i].y - sources[j].y
                    val distSq = dx * dx + dy * dy
                    
                    // phi(r) = r^2 * log(r), with r = sqrt(distSq)
                    // => distSq * log(sqrt(distSq)) = 0.5 * distSq * log(distSq)
                    val ds = max(distSq, 1e-9f)
                    val kVal = if (distSq < 1e-9f) 0f else 0.5f * ds * ln(ds)
                    
                    matL[i][j] = kVal
                    matL[j][i] = kVal
                }
                matL[i][i] += lambda // Regularization
            }

            // Fill P (N x 3) and P^T (3 x N)
            for (i in 0 until n) {
                matL[i][n] = 1f
                matL[i][n + 1] = sources[i].x
                matL[i][n + 2] = sources[i].y

                matL[n][i] = 1f
                matL[n + 1][i] = sources[i].x
                matL[n + 2][i] = sources[i].y
            }

            // 2. Build RHS Y (N+3 x 2)
            // [ V ]
            // [ 0 ]
            val matY = Array(n + 3) { FloatArray(2) }
            for (i in 0 until n) {
                matY[i][0] = targets[i].x
                matY[i][1] = targets[i].y
            }

            // 3. Solve L * X = Y for X (Weights)
            // Using Gaussian Elimination (simple implementation)
            val result = solveLinearSystem(matL, matY)

            // Extract weights and affine part
            weights = FloatArray(n * 2)
            affine = FloatArray(6)

            for (i in 0 until n) {
                weights!![i * 2 + 0] = result[i][0]
                weights!![i * 2 + 1] = result[i][1]
            }
            
            // Affine: a0, a1, a2 for x and y
            // row n -> bias (a0)
            // row n+1 -> coeff x (a1)
            // row n+2 -> coeff y (a2)
            affine!![0] = result[n][0]     // bias X
            affine!![1] = result[n + 1][0] // x coeff X
            affine!![2] = result[n + 2][0] // y coeff X
            
            affine!![3] = result[n][1]     // bias Y
            affine!![4] = result[n + 1][1] // x coeff Y
            affine!![5] = result[n + 2][1] // y coeff Y
        }

        fun transform(gridPoints: List<Point>): List<Point> {
            val cp = controlPoints ?: return emptyList()
            val w = weights ?: return emptyList()
            val aff = affine ?: return emptyList()
            val n = cp.size

            val result = ArrayList<Point>(gridPoints.size)

            for (pt in gridPoints) {
                var sumX = aff[0] + aff[1] * pt.x + aff[2] * pt.y
                var sumY = aff[3] + aff[4] * pt.x + aff[5] * pt.y

                for (i in 0 until n) {
                    val dx = pt.x - cp[i].x
                    val dy = pt.y - cp[i].y
                    val distSq = dx * dx + dy * dy
                    val ds = max(distSq, 1e-9f)
                    val r = if (distSq < 1e-9f) 0f else 0.5f * ds * ln(ds)
                    
                    sumX += r * w[i * 2]
                    sumY += r * w[i * 2 + 1]
                }
                result.add(Point(sumX, sumY))
            }
            return result
        }
        
        // Very basic Gaussian Elimination solver O(N^3)
        // Not optimal for large N, but functional for N=500
        private fun solveLinearSystem(A: Array<FloatArray>, B: Array<FloatArray>): Array<FloatArray> {
            val n = A.size
            val m = B[0].size
            
            // Clone to avoid destroying A
            val mat = Array(n) { A[it].clone() }
            val rhs = Array(n) { B[it].clone() }
            
            val p = IntArray(n) { it } // pivot

            for (i in 0 until n) {
                // Find pivot
                var maxRow = i
                var maxVal = abs(mat[i][i])
                for (k in i + 1 until n) {
                    if (abs(mat[k][i]) > maxVal) {
                        maxVal = abs(mat[k][i])
                        maxRow = k
                    }
                }

                // Swap
                val tempA = mat[i]; mat[i] = mat[maxRow]; mat[maxRow] = tempA
                val tempB = rhs[i]; rhs[i] = rhs[maxRow]; rhs[maxRow] = tempB
                
                // Pivot 0 check
                if (abs(mat[i][i]) < 1e-9f) continue 

                // Eliminate
                for (k in i + 1 until n) {
                    val factor = mat[k][i] / mat[i][i]
                    mat[k][i] = 0f
                    for (j in i + 1 until n) mat[k][j] -= factor * mat[i][j]
                    for (j in 0 until m) rhs[k][j] -= factor * rhs[i][j]
                }
            }

            // Back substitution
            val x = Array(n) { FloatArray(m) }
            for (i in n - 1 downTo 0) {
                if (abs(mat[i][i]) < 1e-9f) continue
                
                for (j in 0 until m) {
                    var sum = rhs[i][j]
                    for (k in i + 1 until n) {
                        sum -= mat[i][k] * x[k][j]
                    }
                    x[i][j] = sum / mat[i][i]
                }
            }
            return x
        }
    }


    fun processImage(originalBitmap: Bitmap, shadingStrength: Float = 1.0f): Bitmap? {
        interpreter ?: return null

        val wOrig = originalBitmap.width
        val hOrig = originalBitmap.height

        // 1. Resize to Model Input Size
        val resized = Bitmap.createScaledBitmap(originalBitmap, INPUT_SIZE, INPUT_SIZE, true)

        // 2. Prepare Input Buffer
        val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * NUM_INPUT_CHANNELS * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Loop and fill (R, G, B, X, Y)
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]
                
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                
                val xx = (x.toFloat() / (INPUT_SIZE - 1)) * 2 - 1
                val yy = (y.toFloat() / (INPUT_SIZE - 1)) * 2 - 1

                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
                inputBuffer.putFloat(xx)
                inputBuffer.putFloat(yy)
            }
        }

        // 3. Run Inference
        val outputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * NUM_OUTPUT_CHANNELS * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // 4. Extract outputs
        val uMap = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val vMap = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val maskMap = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val shadingMap = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)

        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            uMap[i] = outputBuffer.float
            vMap[i] = outputBuffer.float
            maskMap[i] = outputBuffer.float
            shadingMap[i * 3 + 0] = outputBuffer.float
            shadingMap[i * 3 + 1] = outputBuffer.float
            shadingMap[i * 3 + 2] = outputBuffer.float
        }

        // Smooth shading (lighting is low-frequency) similar to vis_unet.py's GaussianBlur on shading_high
        val shadingSmoothed = gaussianBlur3ch(shadingMap, INPUT_SIZE, INPUT_SIZE, radius = 4, sigma = 2.5f)
        val colorCorrectFactor = 0.3f
        val strength = shadingStrength.coerceIn(0f, 1f)

        // 5. Select Control Points for TPS
        // We select points from the Low-Res prediction to fit the TPS
        // Limit number of points to keep solver fast on mobile (N=300-500 is usually enough for smooth paper)
        val MAX_POINTS = 400
        val sources = ArrayList<Point>() // Target UV (u, v)
        val targets = ArrayList<Point>() // Source XY (x, y) normalized 0-1

        var validCount = 0
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            if (maskMap[i] > 0.5f) validCount++
        }
        
        val step = max(1, validCount / MAX_POINTS)
        var counter = 0
        
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val idx = y * INPUT_SIZE + x
                if (maskMap[idx] > 0.5f) {
                    if (counter % step == 0) {
                        val u = uMap[idx]
                        val v = vMap[idx]
                        
                        // Source X,Y normalized 0-1
                        val sx = x.toFloat() / (INPUT_SIZE - 1)
                        val sy = y.toFloat() / (INPUT_SIZE - 1)
                        
                        sources.add(Point(u, v))
                        targets.add(Point(sx, sy))
                    }
                    counter++
                }
            }
        }
        
        if (sources.size < 10) return null // Too few points

        // 6. Fit TPS
        val tps = TPSSolver()
        tps.fit(sources, targets)
        
        // 7. Generate Backward Map (Grid)
        // Keep output dimensions the same as the input image (as requested)
        val targetW = wOrig
        val targetH = hOrig

        // vis_unet.py uses interp_h=512 and resizes. We'll do a smaller intermediate grid for speed,
        // but high enough to look similar (256 px on the long side).
        val GRID_H = 256
        val GRID_W = max(2, (GRID_H * (targetW.toFloat() / targetH.toFloat())).roundToInt())
        
        val gridPoints = ArrayList<Point>(GRID_W * GRID_H)
        for (y in 0 until GRID_H) {
            for (x in 0 until GRID_W) {
                val u = x.toFloat() / (GRID_W - 1)
                val v = y.toFloat() / (GRID_H - 1)
                gridPoints.add(Point(u, v))
            }
        }
        
        val transformedPoints = tps.transform(gridPoints) // Returns Source XY (0-1) for each grid point
        
        // 8. Backward Warp (Remap)
        // We now have a backward map at GRID resolution. We scale it up.
        // We will loop over the FINAL High-Res image pixels.
        // For each pixel (tx, ty), we sample the Backward Map to get (sx, sy).
        // Then we sample the Original Image at (sx, sy).
        
        val resultPixels = IntArray(targetW * targetH)
        val origPixels = IntArray(wOrig * hOrig)
        originalBitmap.getPixels(origPixels, 0, wOrig, 0, 0, wOrig, hOrig)
        
        // Helper for map sampling (Bilinear)
        fun sampleGrid(xNorm: Float, yNorm: Float): Point {
            val gx = xNorm * (GRID_W - 1)
            val gy = yNorm * (GRID_H - 1)
            
            val x0 = gx.toInt().coerceIn(0, GRID_W - 2)
            val y0 = gy.toInt().coerceIn(0, GRID_H - 2)
            val dx = gx - x0
            val dy = gy - y0
            
            val idx00 = y0 * GRID_W + x0
            val idx10 = y0 * GRID_W + (x0 + 1)
            val idx01 = (y0 + 1) * GRID_W + x0
            val idx11 = (y0 + 1) * GRID_W + (x0 + 1)
            
            val p00 = transformedPoints[idx00]
            val p10 = transformedPoints[idx10]
            val p01 = transformedPoints[idx01]
            val p11 = transformedPoints[idx11]
            
            val resX = p00.x * (1 - dx) * (1 - dy) + p10.x * dx * (1 - dy) + p01.x * (1 - dx) * dy + p11.x * dx * dy
            val resY = p00.y * (1 - dx) * (1 - dy) + p10.y * dx * (1 - dy) + p01.y * (1 - dx) * dy + p11.y * dx * dy
            
            return Point(resX, resY)
        }
        
        // Helper to sample Shading Map (Input 256x256)
        // We need shading at SOURCE coordinates to compute Albedo
        fun sampleShading(sxNorm: Float, syNorm: Float): FloatArray {
             val gx = sxNorm * (INPUT_SIZE - 1)
             val gy = syNorm * (INPUT_SIZE - 1)
             
             val x0 = gx.toInt().coerceIn(0, INPUT_SIZE - 2)
             val y0 = gy.toInt().coerceIn(0, INPUT_SIZE - 2)
             val dx = gx - x0
             val dy = gy - y0
             
             val rgb = FloatArray(3)
             for (c in 0..2) {
                 val idx00 = (y0 * INPUT_SIZE + x0) * 3 + c
                 val idx10 = (y0 * INPUT_SIZE + x0 + 1) * 3 + c
                 val idx01 = ((y0 + 1) * INPUT_SIZE + x0) * 3 + c
                 val idx11 = ((y0 + 1) * INPUT_SIZE + x0 + 1) * 3 + c
                 
                 rgb[c] = shadingSmoothed[idx00] * (1 - dx) * (1 - dy) +
                          shadingSmoothed[idx10] * dx * (1 - dy) +
                          shadingSmoothed[idx01] * (1 - dx) * dy +
                          shadingSmoothed[idx11] * dx * dy
             }

             // Luminance mix (color preservation) similar to vis_unet.py
             val lum = (rgb[0] + rgb[1] + rgb[2]) / 3f
             rgb[0] = (1f - colorCorrectFactor) * lum + colorCorrectFactor * rgb[0]
             rgb[1] = (1f - colorCorrectFactor) * lum + colorCorrectFactor * rgb[1]
             rgb[2] = (1f - colorCorrectFactor) * lum + colorCorrectFactor * rgb[2]
             return rgb
        }

        val finalW = targetW
        val finalH = targetH
        val finalPixels = IntArray(finalW * finalH)

        for (y in 0 until finalH) {
            val v = y.toFloat() / (finalH - 1) // Target V 0-1
            
            for (x in 0 until finalW) {
                val u = x.toFloat() / (finalW - 1) // Target U 0-1
                
                // 1. Get Source Coordinates from TPS Grid
                val srcPt = sampleGrid(u, v)
                
                // Check if inside source image
                // Relaxed bounds (like vis_unet.py: [-0.05, 1.05]) to reduce harsh borders
                if (srcPt.x > -0.05f && srcPt.x < 1.05f && srcPt.y > -0.05f && srcPt.y < 1.05f) {
                    
                    // 2. Sample Original Image
                    val sx = (srcPt.x.coerceIn(0f, 1f) * (wOrig - 1)).toInt().coerceIn(0, wOrig - 1)
                    val sy = (srcPt.y.coerceIn(0f, 1f) * (hOrig - 1)).toInt().coerceIn(0, hOrig - 1)
                    
                    val pixel = origPixels[sy * wOrig + sx]
                    
                    val r = ((pixel shr 16) and 0xFF) / 255.0f
                    val g = ((pixel shr 8) and 0xFF) / 255.0f
                    val b = (pixel and 0xFF) / 255.0f
                    
                    // 3. Sample Shading at Source Location (to get Albedo)
                    val shading = sampleShading(srcPt.x.coerceIn(0f, 1f), srcPt.y.coerceIn(0f, 1f))
                    // Apply user strength: 0 -> no correction (denominator=1), 1 -> full correction (denominator=shading)
                    val dR = ((1f - strength) * 1f + strength * shading[0]).coerceAtLeast(0.05f)
                    val dG = ((1f - strength) * 1f + strength * shading[1]).coerceAtLeast(0.05f)
                    val dB = ((1f - strength) * 1f + strength * shading[2]).coerceAtLeast(0.05f)
                    
                    // 4. Calculate Albedo
                    var aR = r / dR
                    var aG = g / dG
                    var aB = b / dB
                    
                    // Simple white balance / clip
                    aR = aR.coerceIn(0f, 1f)
                    aG = aG.coerceIn(0f, 1f)
                    aB = aB.coerceIn(0f, 1f)
                    
                    finalPixels[y * finalW + x] = (0xFF shl 24) or
                            ((aR * 255).toInt() shl 16) or
                            ((aG * 255).toInt() shl 8) or
                            (aB * 255).toInt()
                } else {
                    finalPixels[y * finalW + x] = 0 // Transparent or Black
                }
            }
        }

        val finalBmp = Bitmap.createBitmap(finalW, finalH, Bitmap.Config.ARGB_8888)
        finalBmp.setPixels(finalPixels, 0, finalW, 0, 0, finalW, finalH)

        return finalBmp
    }

    private fun gaussianBlur3ch(src: FloatArray, w: Int, h: Int, radius: Int, sigma: Float): FloatArray {
        if (radius <= 0) return src

        val size = radius * 2 + 1
        val kernel = FloatArray(size)
        var sum = 0f
        for (i in -radius..radius) {
            val v = exp(-(i * i) / (2f * sigma * sigma))
            kernel[i + radius] = v
            sum += v
        }
        for (i in kernel.indices) kernel[i] /= sum

        // Horizontal pass
        val tmp = FloatArray(w * h * 3)
        for (y in 0 until h) {
            for (x in 0 until w) {
                for (c in 0..2) {
                    var acc = 0f
                    for (k in -radius..radius) {
                        val xx = (x + k).coerceIn(0, w - 1)
                        acc += src[(y * w + xx) * 3 + c] * kernel[k + radius]
                    }
                    tmp[(y * w + x) * 3 + c] = acc
                }
            }
        }

        // Vertical pass
        val out = FloatArray(w * h * 3)
        for (y in 0 until h) {
            for (x in 0 until w) {
                for (c in 0..2) {
                    var acc = 0f
                    for (k in -radius..radius) {
                        val yy = (y + k).coerceIn(0, h - 1)
                        acc += tmp[(yy * w + x) * 3 + c] * kernel[k + radius]
                    }
                    out[(y * w + x) * 3 + c] = acc
                }
            }
        }
        return out
    }
}
