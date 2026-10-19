package com.example.uvdoc

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Runs the backward-map model and dewarps the photo.
 *
 * Model output: backward map (1, GRID_H, GRID_W, 2), NHWC. Entry (i, j) is the position (x, y) in the input
 * image of the flat-page point (u = j / (GRID_W - 1), v = i / (GRID_H - 1)), normalized to [-1, 1]
 * (align_corners semantics). Dewarping = bilinear lookup in the grid + bilinear sampling of the photo.
 */
class ImageProcessor(context: Context) {

    private var modelBuffer: ByteBuffer? = null
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private val modelName = "model_fp16.tflite"
    var loadError: String? = null
        private set

    private val INPUT_SIZE = 256
    private val NUM_INPUT_CHANNELS = 5 // R, G, B, X, Y
    private val GRID_H = 32
    private val GRID_W = 32
    private val A4_RATIO = 1.414f

    init {
        try {
            val fd = context.assets.openFd(modelName)
            modelBuffer = java.io.FileInputStream(fd.fileDescriptor).channel
                .map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        } catch (e: Exception) {
            loadError = e.message ?: e.toString()
            Log.e(TAG, "Failed to open $modelName", e)
        }
    }

    /** Created lazily: the GPU delegate must run on the thread that created it (the worker thread). */
    private fun getInterpreter(): Interpreter? {
        if (interpreter == null) {
            interpreter = modelBuffer?.let { createInterpreter(it) }
        }
        return interpreter
    }

    private fun createInterpreter(model: ByteBuffer): Interpreter {
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            try {
                val delegate = GpuDelegate()
                val interp = Interpreter(model, Interpreter.Options().addDelegate(delegate))
                gpuDelegate = delegate
                Log.i(TAG, "Using GPU delegate")
                return interp
            } catch (e: Throwable) {
                Log.w(TAG, "GPU delegate failed, falling back to CPU", e)
                gpuDelegate?.close()
                gpuDelegate = null
            }
        }
        Log.i(TAG, "Using CPU (XNNPACK)")
        return Interpreter(model, Interpreter.Options().setUseXNNPACK(true).setNumThreads(4))
    }

    /** Call on the same thread as processImage. */
    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
    }

    private fun prepareInput(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * NUM_INPUT_CHANNELS * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (y in 0 until INPUT_SIZE) {
            val yy = y.toFloat() / (INPUT_SIZE - 1) * 2 - 1
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                buffer.putFloat((pixel and 0xFF) / 255.0f)
                buffer.putFloat(x.toFloat() / (INPUT_SIZE - 1) * 2 - 1)
                buffer.putFloat(yy)
            }
        }
        buffer.rewind()
        return buffer
    }

    /** Returns the backward map as interleaved (x, y) floats, row-major over the GRID_H x GRID_W grid. */
    private fun predictGrid(bitmap: Bitmap): FloatArray? {
        val interp = getInterpreter()
        if (interp == null) {
            if (loadError == null) loadError = "model buffer is null"
            return null
        }
        val grid = try {
            readGrid(interp, bitmap)
        } catch (e: Exception) {
            loadError = e.message ?: e.toString()
            Log.e(TAG, "Interpreter run failed", e)
            return null
        }
        if (gpuDelegate != null && grid.any { !it.isFinite() }) {
            Log.w(TAG, "GPU backward map is non-finite, retrying on CPU")
            switchToCpu()
            val cpu = interpreter ?: return null
            return try {
                readGrid(cpu, bitmap)
            } catch (e: Exception) {
                loadError = e.message ?: e.toString()
                Log.e(TAG, "CPU retry failed", e)
                null
            }
        }
        return grid
    }

    private fun readGrid(interp: Interpreter, bitmap: Bitmap): FloatArray {
        val output = ByteBuffer.allocateDirect(GRID_H * GRID_W * 2 * 4).order(ByteOrder.nativeOrder())
        interp.run(prepareInput(bitmap), output)
        output.rewind()
        val grid = FloatArray(GRID_H * GRID_W * 2)
        output.asFloatBuffer().get(grid)
        return grid
    }

    /** Replaces a bad GPU interpreter. Must run on the worker thread. */
    private fun switchToCpu() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
        val model = modelBuffer ?: return
        interpreter = Interpreter(model, Interpreter.Options().setUseXNNPACK(true).setNumThreads(4))
        Log.i(TAG, "Using CPU (XNNPACK)")
    }

    fun processImage(originalBitmap: Bitmap): Bitmap? {
        val grid = predictGrid(originalBitmap) ?: return null

        val wOrig = originalBitmap.width
        val hOrig = originalBitmap.height
        val origPixels = IntArray(wOrig * hOrig)
        originalBitmap.getPixels(origPixels, 0, wOrig, 0, 0, wOrig, hOrig)

        // Portrait A4 output at the resolution of the longer input side
        val outH = max(wOrig, hOrig)
        val outW = max(2, (outH / A4_RATIO).roundToInt())
        val outPixels = IntArray(outW * outH)

        val sx = (wOrig - 1) * 0.5f
        val sy = (hOrig - 1) * 0.5f

        for (y in 0 until outH) {
            // Row position in the grid (align_corners)
            val gy = y.toFloat() / (outH - 1) * (GRID_H - 1)
            val y0 = gy.toInt().coerceAtMost(GRID_H - 2)
            val fy = gy - y0

            for (x in 0 until outW) {
                val gx = x.toFloat() / (outW - 1) * (GRID_W - 1)
                val x0 = gx.toInt().coerceAtMost(GRID_W - 2)
                val fx = gx - x0

                val i00 = (y0 * GRID_W + x0) * 2
                val i10 = i00 + 2
                val i01 = i00 + GRID_W * 2
                val i11 = i01 + 2
                val w00 = (1 - fx) * (1 - fy)
                val w10 = fx * (1 - fy)
                val w01 = (1 - fx) * fy
                val w11 = fx * fy

                // Normalized [-1, 1] source position -> pixel coordinates
                val srcX = (grid[i00] * w00 + grid[i10] * w10 + grid[i01] * w01 + grid[i11] * w11 + 1f) * sx
                val srcY = (grid[i00 + 1] * w00 + grid[i10 + 1] * w10 + grid[i01 + 1] * w01 + grid[i11 + 1] * w11 + 1f) * sy

                outPixels[y * outW + x] = sampleBilinear(origPixels, wOrig, hOrig, srcX, srcY)
            }
        }

        val result = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        result.setPixels(outPixels, 0, outW, 0, 0, outW, outH)
        return result
    }

    /** Bilinear RGB sample; black outside the image. */
    private fun sampleBilinear(pixels: IntArray, w: Int, h: Int, x: Float, y: Float): Int {
        if (!x.isFinite() || !y.isFinite() || x < 0f || y < 0f || x > w - 1 || y > h - 1) {
            return 0xFF000000.toInt()
        }
        val x0 = x.toInt().coerceAtMost(w - 2)
        val y0 = y.toInt().coerceAtMost(h - 2)
        val fx = x - x0
        val fy = y - y0

        val p00 = pixels[y0 * w + x0]
        val p10 = pixels[y0 * w + x0 + 1]
        val p01 = pixels[(y0 + 1) * w + x0]
        val p11 = pixels[(y0 + 1) * w + x0 + 1]

        fun channel(shift: Int): Int {
            val c00 = (p00 shr shift) and 0xFF
            val c10 = (p10 shr shift) and 0xFF
            val c01 = (p01 shr shift) and 0xFF
            val c11 = (p11 shr shift) and 0xFF
            val top = c00 + (c10 - c00) * fx
            val bottom = c01 + (c11 - c01) * fx
            return (top + (bottom - top) * fy).roundToInt().coerceIn(0, 255)
        }

        return (0xFF shl 24) or (channel(16) shl 16) or (channel(8) shl 8) or channel(0)
    }

    companion object {
        private const val TAG = "ImageProcessor"
    }
}
