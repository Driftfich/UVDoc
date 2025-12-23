package com.example.uvdoc

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var btnLoad: Button
    private lateinit var btnCamera: Button
    private lateinit var btnProcess: Button
    private lateinit var imageView: ImageView
    private lateinit var previewView: PreviewView
    private lateinit var progressBar: ProgressBar
    private lateinit var seekStrength: SeekBar
    private lateinit var txtStrength: TextView

    private var currentBitmap: Bitmap? = null
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var cameraExecutor: ExecutorService
    
    private var imageCapture: ImageCapture? = null
    private var shadingStrength: Float = 1.0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnLoad = findViewById(R.id.btnLoad)
        btnCamera = findViewById(R.id.btnCamera)
        btnProcess = findViewById(R.id.btnProcess)
        imageView = findViewById(R.id.imageView)
        previewView = findViewById(R.id.viewFinder)
        progressBar = findViewById(R.id.progressBar)
        seekStrength = findViewById(R.id.seekStrength)
        txtStrength = findViewById(R.id.txtStrength)

        imageProcessor = ImageProcessor(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Slider: 0..100 -> 0.00..1.00
        shadingStrength = seekStrength.progress / 100f
        txtStrength.text = "Shading strength: %.2f".format(shadingStrength)
        seekStrength.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                shadingStrength = progress / 100f
                txtStrength.text = "Shading strength: %.2f".format(shadingStrength)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        btnLoad.setOnClickListener {
            openGallery()
        }

        btnCamera.setOnClickListener {
            if (previewView.visibility == View.VISIBLE) {
                takePhoto()
            } else {
                if (allPermissionsGranted()) {
                    startCamera()
                } else {
                    ActivityCompat.requestPermissions(
                        this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
                    )
                }
            }
        }

        btnProcess.setOnClickListener {
            processImage()
        }
    }

    private fun allPermissionsGranted() : Boolean {
        for (permission in REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(baseContext, permission) != PackageManager.PERMISSION_GRANTED) {
                // Special case: some older devices might report false for storage but we only NEED camera to start the preview
                if (permission == Manifest.permission.CAMERA) return false
            }
        }
        return true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA
            ).toTypedArray()
    }

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                loadBitmapFromUri(uri)
            }
        }
    }

    private fun openGallery() {
        stopCamera()
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        galleryLauncher.launch(intent)
    }

    private fun loadBitmapFromUri(uri: Uri) {
        val bitmap = contentResolver.openInputStream(uri).use { input ->
            if (input == null) return
            BitmapFactory.decodeStream(input)
        } ?: return
        var rotatedBitmap = bitmap
        
        // Handle Rotation via EXIF
        try {
            val orientation = contentResolver.openInputStream(uri).use { exifStream ->
                if (exifStream == null) ExifInterface.ORIENTATION_NORMAL
                else ExifInterface(exifStream).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            }
                
            var rotation = 0f
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotation = 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> rotation = 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> rotation = 270f
            }
                
            if (rotation != 0f) {
                val matrix = Matrix()
                matrix.postRotate(rotation)
                rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        
        currentBitmap = rotatedBitmap
        imageView.setImageBitmap(currentBitmap)
        imageView.visibility = View.VISIBLE
        previewView.visibility = View.GONE
        btnProcess.isEnabled = true
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val targetRotation = previewView.display?.rotation ?: android.view.Surface.ROTATION_0
            val preview = Preview.Builder()
                .setTargetRotation(targetRotation)
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetRotation(targetRotation)
                .build()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )

                // Only show preview if binding succeeded
                imageView.visibility = View.GONE
                previewView.visibility = View.VISIBLE
                btnCamera.text = "Click to Capture"

            } catch(exc: Exception) {
                Toast.makeText(this, "Use case binding failed: ${exc.message}", Toast.LENGTH_SHORT).show()
            }

        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun stopCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()
        }, ContextCompat.getMainExecutor(this))
        previewView.visibility = View.GONE
        btnCamera.text = getString(R.string.take_photo)
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        // Use internal cache to avoid needing any external storage permissions
        val photoFile = File(cacheDir, "temp_photo.jpg")

        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Ensure capture rotation matches preview rotation
        imageCapture.targetRotation = previewView.display?.rotation ?: android.view.Surface.ROTATION_0

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(baseContext, "Photo failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    loadBitmapFromUri(savedUri)
                    stopCamera()
                }
            }
        )
    }

    private fun processImage() {
        val bitmap = currentBitmap ?: return
        
        progressBar.visibility = View.VISIBLE
        btnProcess.isEnabled = false
        
        cameraExecutor.execute {
            val startTime = System.currentTimeMillis()
            val result = imageProcessor.processImage(bitmap, shadingStrength)
            val time = System.currentTimeMillis() - startTime
            
            runOnUiThread {
                progressBar.visibility = View.GONE
                btnProcess.isEnabled = true
                if (result != null) {
                    imageView.setImageBitmap(result)
                    currentBitmap = result // So we can't process again recursively
                    Toast.makeText(this, "Processed in ${time}ms", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Processing failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

