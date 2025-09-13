
#Updated Kotlin API: FaceSDK.kt


package com.neptune.sdk

import android.content.Context
import android.graphics.Bitmap
import java.io.File

class FaceSDK private constructor(private val sdkPtr: Long) {
    
    companion object {
        init {
            System.loadLibrary("neptune_sdk")
            System.loadLibrary("neptune_core")
            // Load other native dependencies (OpenCV, TFLite)
        }
        
        @JvmStatic
        external fun nativeCreate(
            faceModelPath: String,
            emotionModelPath: String,
            livenessModelPath: String,
            minFaceConfidence: Float,
            minEmotionConfidence: Float,
            processingWidth: Int,
            processingHeight: Int
        ): Long
        
        @JvmStatic
        external fun nativeProcessImage(
            sdkPtr: Long,
            imageData: ByteArray,
            width: Int,
            height: Int
        ): Array<ProcessingResult>?
        
        @JvmStatic
        external fun nativeRelease(sdkPtr: Long)
    }
    
    fun processImage(bitmap: Bitmap): Array<ProcessingResult>? {
        val byteArray = bitmapToByteArray(bitmap)
        return nativeProcessImage(sdkPtr, byteArray, bitmap.width, bitmap.height)
    }
    
    fun processImage(imageData: ByteArray, width: Int, height: Int): Array<ProcessingResult>? {
        return nativeProcessImage(sdkPtr, imageData, width, height)
    }
    
    fun release() {
        nativeRelease(sdkPtr)
    }
    
    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val byteCount = bitmap.byteCount
        val buffer = java.nio.ByteBuffer.allocate(byteCount)
        bitmap.copyPixelsToBuffer(buffer)
        return buffer.array()
    }
    
    class Builder(private val context: Context) {
        private var minFaceConfidence: Float = 0.5f
        private var minEmotionConfidence: Float = 0.2f
        private var processingWidth: Int = 320
        private var processingHeight: Int = 240
        
        fun minFaceConfidence(confidence: Float): Builder {
            this.minFaceConfidence = confidence
            return this
        }
        
        fun minEmotionConfidence(confidence: Float): Builder {
            this.minEmotionConfidence = confidence
            return this
        }
        
        fun processingSize(width: Int, height: Int): Builder {
            this.processingWidth = width
            this.processingHeight = height
            return this
        }
        
        fun build(): FaceSDK {
            val modelDir = copyModelsToInternalStorage(context)
            val sdkPtr = nativeCreate(
                "${modelDir.absolutePath}/face_detection.tflite",
                "${modelDir.absolutePath}/emotion_model.tflite",
                "${modelDir.absolutePath}/liveness_model.tflite",
                minFaceConfidence,
                minEmotionConfidence,
                processingWidth,
                processingHeight
            )
            
            if (sdkPtr == 0L) {
                throw RuntimeException("Failed to initialize Neptune SDK")
            }
            
            return FaceSDK(sdkPtr)
        }
        
        private fun copyModelsToInternalStorage(context: Context): File {
            val modelDir = File(context.filesDir, "neptune_models")
            if (!modelDir.exists()) {
                modelDir.mkdirs()
                
                // Copy models from assets
                arrayOf("face_detection", "emotion_model", "liveness_model").forEach { modelName ->
                    val inputStream = context.assets.open("models/$modelName.tflite")
                    val outputFile = File(modelDir, "$modelName.tflite")
                    inputStream.use { input ->
                        FileOutputStream(outputFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                }
            }
            return modelDir
        }
    }
}