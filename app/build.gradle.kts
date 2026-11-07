plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.liquid.lfm3"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.liquid.lfm3"
        minSdk = 31
        targetSdk = 35
        versionCode = 4
        versionName = "1.2.0"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file("../keystore.jks")
            storePassword = "lfm3npu2026"
            keyAlias = "lfm3"
            keyPassword = "lfm3npu2026"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
        debug {
            signingConfig = signingConfigs.getByName("release")
        }
    }

    buildFeatures {
        buildConfig = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    packaging {
        resources.excludes += setOf(
            "META-INF/AL2.0", "META-INF/LGPL2.1", "META-INF/DEPENDENCIES",
            "META-INF/LICENSE.md", "META-INF/LICENSE-notice.md"
        )
    }
}

dependencies {
    // ONNX Runtime mit QNN Execution Provider (Hexagon NPU) – bringt qnn-runtime transitiv mit
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Lokaler HTTP-Server für die HTML-Apps (OpenAI-kompatible API)
    implementation("org.nanohttpd:nanohttpd:2.3.1")
}
