package com.example.uvdoc

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class ImageProcessor(context: Context) {

    private var interpreter: Interpreter? = null
    private val modelName = "model.tflite" // Name ggf. anpassen

    // Model dims
    private val INPUT_SIZE = 256
    private val NUM_INPUT_CHANNELS = 5 // R, G, B, X, Y
    private val NUM_OUTPUT_CHANNELS = 3 // U, V, Mask (Shading entfernt)

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

    data class Point(val x: Float, val y: Float)

    class TPSSolver {
        private var weights: FloatArray? = null
        private var affine: FloatArray? = null
        private var controlPoints: List<Point>? = null
        private var lambda = 0.01f

        fun fit(sources: List<Point>, targets: List<Point>) {
            val n = sources.size
            if (n == 0) return

            controlPoints = sources

            val matL = Array(n + 3) { FloatArray(n + 3) }

            for (i in 0 until n) {
                for (j in i until n) {
                    val dx = sources[i].x - sources[j].x
                    val dy = sources[i].y - sources[j].y
                    val distSq = dx * dx + dy * dy
                    
                    val ds = max(distSq, 1e-9f)
                    val kVal = if (distSq < 1e-9f) 0f else 0.5f * ds * ln(ds)
                    
                    matL[i][j] = kVal
                    matL[j][i] = kVal
                }
                matL[i][i] += lambda
            }

            for (i in 0 until n) {
                matL[i][n] = 1f
                matL[i][n + 1] = sources[i].x
                matL[i][n + 2] = sources[i].y

                matL[n][i] = 1f
                matL[n + 1][i] = sources[i].x
                matL[n + 2][i] = sources[i].y
            }

            val matY = Array(n + 3) { FloatArray(2) }
            for (i in 0 until n) {
                matY[i][0] = targets[i].x
                matY[i][1] = targets[i].y
            }

            val result = solveLinearSystem(matL, matY)

            weights = FloatArray(n * 2)
            affine = FloatArray(6)

            for (i in 0 until n) {
                weights!![i * 2 + 0] = result[i][0]
                weights!![i * 2 + 1] = result[i][1]
            }
            
            affine!![0] = result[n][0]     
            affine!![1] = result[n + 1][0] 
            affine!![2] = result[n + 2][0] 
            
            affine!![3] = result[n][1]     
            affine!![4] = result[n + 1][1] 
            affine!![5] = result[n + 2][1] 
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
        
        private fun solveLinearSystem(A: Array<FloatArray>, B: Array<FloatArray>): Array<FloatArray> {
            val n = A.size
            val m = B[0].size
            val mat = Array(n) { A[it].clone() }
            val rhs = Array(n) { B[it].clone() }
            
            for (i in 0 until n) {
                var maxRow = i
                var maxVal = abs(mat[i][i])
                for (k in i + 1 until n) {
                    if (abs(mat[k][i]) > maxVal) {
                        maxVal = abs(mat[k][i])
                        maxRow = k
                    }
                }

                val tempA = mat[i]; mat[i] = mat[maxRow]; mat[maxRow] = tempA
                val tempB = rhs[i]; rhs[i] = rhs[maxRow]; rhs[maxRow] = tempB
                
                if (abs(mat[i][i]) < 1e-9f) continue 

                for (k in i + 1 until n) {
                    val factor = mat[k][i] / mat[i][i]
                    mat[k][i] = 0f
                    for (j in i + 1 until n) mat[k][j] -= factor * mat[i][j]
                    for (j in 0 until m) rhs[k][j] -= factor * rhs[i][j]
                }
            }

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


    fun processImage(originalBitmap: Bitmap): Bitmap? {
        interpreter ?: return null

        val wOrig = originalBitmap.width
        val hOrig = originalBitmap.height

        val resized = Bitmap.createScaledBitmap(originalBitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * NUM_INPUT_CHANNELS * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

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

        val outputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * NUM_OUTPUT_CHANNELS * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val uMap = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val vMap = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val maskMap = FloatArray(INPUT_SIZE * INPUT_SIZE)

        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            uMap[i] = outputBuffer.float
            vMap[i] = outputBuffer.float
            maskMap[i] = outputBuffer.float
        }

        // --- NEW SIGMOID FIX ---
        var minMask = Float.MAX_VALUE
        var maxMask = -Float.MAX_VALUE
        for (m in maskMap) {
            if (m < minMask) minMask = m
            if (m > maxMask) maxMask = m
        }

        // Apply Sigmoid if raw logits are detected
        if (minMask < -0.1f || maxMask > 1.1f) {
            for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
                uMap[i] = (1.0 / (1.0 + exp(-uMap[i].toDouble()))).toFloat()
                vMap[i] = (1.0 / (1.0 + exp(-vMap[i].toDouble()))).toFloat()
                maskMap[i] = (1.0 / (1.0 + exp(-maskMap[i].toDouble()))).toFloat()
            }
        }
        // -----------------------

        val threshold = 0.5f
        val MAX_POINTS = 1200
        val sources = ArrayList<Point>() 
        val targets = ArrayList<Point>() 

        var validCount = 0
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            if (maskMap[i] > threshold) validCount++
        }
        
        val step = max(1, validCount / MAX_POINTS)
        var counter = 0
        
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val idx = y * INPUT_SIZE + x
                if (maskMap[idx] > threshold) {
                    if (counter % step == 0) {
                        val u = uMap[idx]
                        val v = vMap[idx]
                        
                        val sx = x.toFloat() / (INPUT_SIZE - 1)
                        val sy = y.toFloat() / (INPUT_SIZE - 1)
                        
                        sources.add(Point(u, v))
                        targets.add(Point(sx, sy))
                    }
                    counter++
                }
            }
        }
        
        if (sources.size < 10) return null

        val tps = TPSSolver()
        tps.fit(sources, targets)
        
        val targetW = wOrig
        val targetH = hOrig

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
        
        val transformedPoints = tps.transform(gridPoints) 
        
        val origPixels = IntArray(wOrig * hOrig)
        originalBitmap.getPixels(origPixels, 0, wOrig, 0, 0, wOrig, hOrig)
        
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

        val finalPixels = IntArray(targetW * targetH)

        for (y in 0 until targetH) {
            val v = y.toFloat() / (targetH - 1) 
            
            for (x in 0 until targetW) {
                val u = x.toFloat() / (targetW - 1) 
                
                val srcPt = sampleGrid(u, v)
                
                if (srcPt.x > -0.05f && srcPt.x < 1.05f && srcPt.y > -0.05f && srcPt.y < 1.05f) {
                    val sx = (srcPt.x.coerceIn(0f, 1f) * (wOrig - 1)).toInt().coerceIn(0, wOrig - 1)
                    val sy = (srcPt.y.coerceIn(0f, 1f) * (hOrig - 1)).toInt().coerceIn(0, hOrig - 1)
                    
                    finalPixels[y * targetW + x] = origPixels[sy * wOrig + sx]
                } else {
                    finalPixels[y * targetW + x] = 0
                }
            }
        }

        val finalBmp = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        finalBmp.setPixels(finalPixels, 0, targetW, 0, 0, targetW, targetH)

        return finalBmp
    }
}