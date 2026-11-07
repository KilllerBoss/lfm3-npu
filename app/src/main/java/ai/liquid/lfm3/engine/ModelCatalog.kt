package ai.liquid.lfm3.engine

/**
 * Statischer Modellkatalog aller LiquidAI LFM3 ONNX-Varianten.
 *
 * Der Block zwischen GENERATED-CATALOG-BEGIN/END wird von
 * scripts/gen_catalog.py aus den echten HuggingFace-Dateilisten
 * (tree.json) mit exakten Dateipfaden und Größen neu geschrieben.
 */

enum class ModelKind(val label: String) {
    TEXT("Text"),
    VISION("Vision"),
    AUDIO("Audio"),
    OMNI("Omni"),
    DECISION("Decision/JSON")
}

/** Eine Datei im HF-Repo mit exakter Größe in Bytes. */
data class FileSpec(val path: String, val bytes: Long)

/** Eine Quantisierungs-/Präzisionsvariante eines Modells. */
data class PrecisionInfo(
    val id: String,            // "q4", "q4f32", "q8", "q4f16", "q8f16", "fp16", "fp32"
    val decoderPath: String,   // z.B. "onnx/model_q4.onnx"
    val visionPath: String?,   // nur für VL/Omni: eigener Vision-Encoder
    val files: List<FileSpec>  // alle zum decoder/vision gehörenden Dateien inkl. Größen
)

/** Ein Modell (HF-Repo) mit allen verfügbaren Präzisionen. */
data class ModelInfo(
    val id: String,                  // "LFM3-1.7B-ONNX"
    val repo: String,                // "LiquidAI/LFM3-1.7B-ONNX"
    val displayName: String,
    val kind: ModelKind,
    val desc: String,
    val needsVision: Boolean,        // Bild-Eingabe im Decoder
    val commonFiles: List<FileSpec>, // tokenizer.json etc. (pro Modell, einmalig)
    val precisions: List<PrecisionInfo>
) {
    fun precision(precId: String): PrecisionInfo? = precisions.firstOrNull { it.id == precId }

    fun totalBytes(precId: String): Long {
        val p = precision(precId) ?: return 0
        return commonFiles.sumOf { it.bytes } + p.files.sumOf { it.bytes }
    }
}

object ModelCatalog {

    const val DEFAULT_PRECISION = "q4"

    fun defaultModel(): ModelInfo = MODELS.first { it.id == "LFM3-1.7B-ONNX" }

    fun byId(id: String): ModelInfo? = MODELS.firstOrNull { it.id == id }

    // ===== GENERATED-CATALOG-BEGIN =====
    // Automatisch erzeugt von scripts/gen_catalog.py aus HF tree.json
    val MODELS: List<ModelInfo> = listOf(
        ModelInfo(
            id = "LFM3-1.7B-ONNX",
            repo = "LiquidAI/LFM3-1.7B-ONNX",
            displayName = "LFM3 1.7B (Text)",
            kind = ModelKind.TEXT,
            desc = "Kompaktes Text-Chat-Modell, sehr schnell auf dem NPU.",
            needsVision = false,
            commonFiles = listOf(
                FileSpec("tokenizer.json", 17908634L),
                FileSpec("tokenizer_config.json", 14588L),
                FileSpec("chat_template.jinja", 12143L),
                FileSpec("config.json", 1657L),
                FileSpec("generation_config.json", 146L)
            ),
            precisions = listOf(
                PrecisionInfo("q4", "onnx/model_q4.onnx", null,
                    listOf(
                        FileSpec("onnx/model_q4.onnx", 352630L),
                        FileSpec("onnx/model_q4.onnx_data", 1625944064L)
                    )),
                PrecisionInfo("q4f32", "onnx/model_q4f32.onnx", null,
                    listOf(
                        FileSpec("onnx/model_q4f32.onnx", 322164L),
                        FileSpec("onnx/model_q4f32.onnx_data", 2147131392L),
                        FileSpec("onnx/model_q4f32.onnx_data_1", 87838720L)
                    )),
                PrecisionInfo("q8", "onnx/model_q8.onnx", null,
                    listOf(
                        FileSpec("onnx/model_q8.onnx", 347378L),
                        FileSpec("onnx/model_q8.onnx_data", 1965349888L),
                        FileSpec("onnx/model_q8.onnx_data_1", 303104000L)
                    )),
                PrecisionInfo("q4f16", "onnx/model_q4f16.onnx", null,
                    listOf(
                        FileSpec("onnx/model_q4f16.onnx", 268082L),
                        FileSpec("onnx/model_q4f16.onnx_data", 1382930432L)
                    )),
                PrecisionInfo("q8f16", "onnx/model_q8f16.onnx", null,
                    listOf(
                        FileSpec("onnx/model_q8f16.onnx", 262920L),
                        FileSpec("onnx/model_q8f16.onnx_data", 2025440256L)
                    )),
                PrecisionInfo("fp16", "onnx/model_fp16.onnx", null,
                    listOf(
                        FileSpec("onnx/model_fp16.onnx", 270733L),
                        FileSpec("onnx/model_fp16.onnx_data", 2131447808L),
                        FileSpec("onnx/model_fp16.onnx_data_1", 1460367360L)
                    )),
                PrecisionInfo("fp32", "onnx/model.onnx", null,
                    listOf(
                        FileSpec("onnx/model.onnx", 274961L),
                        FileSpec("onnx/model.onnx_data", 2122694656L),
                        FileSpec("onnx/model.onnx_data_1", 2140200960L),
                        FileSpec("onnx/model.onnx_data_2", 2135916544L),
                        FileSpec("onnx/model.onnx_data_3", 784818176L)
                    ))
            )
        ),
        ModelInfo(
            id = "LFM3-VL-2B-ONNX",
            repo = "LiquidAI/LFM3-VL-2B-ONNX",
            displayName = "LFM3 VL 2B (Vision)",
            kind = ModelKind.VISION,
            desc = "Bildverständnis + Chat: Fotos beschreiben, Fragen zu Bildern.",
            needsVision = true,
            commonFiles = listOf(
                FileSpec("tokenizer.json", 17908258L),
                FileSpec("tokenizer_config.json", 1759L),
                FileSpec("chat_template.jinja", 12143L),
                FileSpec("processor_config.json", 1535L),
                FileSpec("config.json", 2905L),
                FileSpec("generation_config.json", 355L)
            ),
            precisions = listOf(
                PrecisionInfo("q4", "onnx/decoder_q4.onnx", "onnx/vision_encoder_q8.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q4.onnx", 357620L),
                        FileSpec("onnx/decoder_q4.onnx_data", 1488580608L),
                        FileSpec("onnx/vision_encoder_q8.onnx", 126353L),
                        FileSpec("onnx/vision_encoder_q8.onnx_data", 196153344L)
                    )),
                PrecisionInfo("q8", "onnx/decoder_q8.onnx", "onnx/vision_encoder_q8.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q8.onnx", 352437L),
                        FileSpec("onnx/decoder_q8.onnx_data", 2131090432L),
                        FileSpec("onnx/vision_encoder_q8.onnx", 126353L),
                        FileSpec("onnx/vision_encoder_q8.onnx_data", 196153344L)
                    )),
                PrecisionInfo("q4f16", "onnx/decoder_q4f16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q4f16.onnx", 273090L),
                        FileSpec("onnx/decoder_q4f16.onnx_data", 1314248704L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("q8f16", "onnx/decoder_q8f16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q8f16.onnx", 267969L),
                        FileSpec("onnx/decoder_q8f16.onnx_data", 1956758528L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("fp16", "onnx/decoder_fp16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_fp16.onnx", 275427L),
                        FileSpec("onnx/decoder_fp16.onnx_data", 2125680640L),
                        FileSpec("onnx/decoder_fp16.onnx_data_1", 1397452800L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("fp32", "onnx/decoder.onnx", "onnx/vision_encoder.onnx",
                    listOf(
                        FileSpec("onnx/decoder.onnx", 279435L),
                        FileSpec("onnx/decoder.onnx_data", 2111160320L),
                        FileSpec("onnx/decoder.onnx_data_1", 2140200960L),
                        FileSpec("onnx/decoder.onnx_data_2", 2115108864L),
                        FileSpec("onnx/decoder.onnx_data_3", 679796736L),
                        FileSpec("onnx/vision_encoder.onnx", 105627L),
                        FileSpec("onnx/vision_encoder.onnx_data", 758042624L)
                    ))
            )
        ),
        ModelInfo(
            id = "LFM3-Audio-2B-ONNX",
            repo = "LiquidAI/LFM3-Audio-2B-ONNX",
            displayName = "LFM3 Audio 2B",
            kind = ModelKind.AUDIO,
            desc = "Audio-Modell; Sprach-Ein/-Ausgabe läuft über System-Speech.",
            needsVision = false,
            commonFiles = listOf(
                FileSpec("tokenizer.json", 17908634L),
                FileSpec("tokenizer_config.json", 674L),
                FileSpec("chat_template.jinja", 9342L),
                FileSpec("processor_config.json", 773L),
                FileSpec("preprocessor_config.json", 466L),
                FileSpec("special_tokens_map.json", 92L),
                FileSpec("config.json", 6243L),
                FileSpec("generation_config.json", 382L)
            ),
            precisions = listOf(
                PrecisionInfo("q4", "onnx/decoder_q4.onnx", null,
                    listOf(
                        FileSpec("onnx/decoder_q4.onnx", 358840L),
                        FileSpec("onnx/decoder_q4.onnx_data", 1686859776L)
                    )),
                PrecisionInfo("q8", "onnx/decoder_q8.onnx", null,
                    listOf(
                        FileSpec("onnx/decoder_q8.onnx", 353613L),
                        FileSpec("onnx/decoder_q8.onnx_data", 2026265600L),
                        FileSpec("onnx/decoder_q8.onnx_data_1", 303104000L)
                    )),
                PrecisionInfo("q4f16", "onnx/decoder_q4f16.onnx", null,
                    listOf(
                        FileSpec("onnx/decoder_q4f16.onnx", 274325L),
                        FileSpec("onnx/decoder_q4f16.onnx_data", 1413388288L)
                    )),
                PrecisionInfo("q8f16", "onnx/decoder_q8f16.onnx", null,
                    listOf(
                        FileSpec("onnx/decoder_q8f16.onnx", 269173L),
                        FileSpec("onnx/decoder_q8f16.onnx_data", 2055898112L)
                    )),
                PrecisionInfo("fp16", "onnx/decoder_fp16.onnx", null,
                    listOf(
                        FileSpec("onnx/decoder_fp16.onnx", 276580L),
                        FileSpec("onnx/decoder_fp16.onnx_data", 2128306176L),
                        FileSpec("onnx/decoder_fp16.onnx_data_1", 1493966848L)
                    )),
                PrecisionInfo("fp32", "onnx/decoder.onnx", null,
                    listOf(
                        FileSpec("onnx/decoder.onnx", 280541L),
                        FileSpec("onnx/decoder.onnx_data", 2120695808L),
                        FileSpec("onnx/decoder.onnx_data_1", 2135916544L),
                        FileSpec("onnx/decoder.onnx_data_2", 2140200960L),
                        FileSpec("onnx/decoder.onnx_data_3", 847732736L)
                    ))
            )
        ),
        ModelInfo(
            id = "LFM3-Omni-2B-ONNX",
            repo = "LiquidAI/LFM3-Omni-2B-ONNX",
            displayName = "LFM3 Omni 2B (Vision+Audio)",
            kind = ModelKind.OMNI,
            desc = "Text + Bild + Sprache in einem Modell.",
            needsVision = true,
            commonFiles = listOf(
                FileSpec("tokenizer.json", 17908634L),
                FileSpec("tokenizer_config.json", 674L),
                FileSpec("chat_template.jinja", 13527L),
                FileSpec("processor_config.json", 2374L),
                FileSpec("preprocessor_config.json", 536L),
                FileSpec("config.json", 5912L),
                FileSpec("generation_config.json", 392L)
            ),
            precisions = listOf(
                PrecisionInfo("q4", "onnx/decoder_q4.onnx", "onnx/vision_encoder_q8.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q4.onnx", 362612L),
                        FileSpec("onnx/decoder_q4.onnx_data", 1686859776L),
                        FileSpec("onnx/vision_encoder_q8.onnx", 126353L),
                        FileSpec("onnx/vision_encoder_q8.onnx_data", 196153344L)
                    )),
                PrecisionInfo("q8", "onnx/decoder_q8.onnx", "onnx/vision_encoder_q8.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q8.onnx", 357385L),
                        FileSpec("onnx/decoder_q8.onnx_data", 2026265600L),
                        FileSpec("onnx/decoder_q8.onnx_data_1", 303104000L),
                        FileSpec("onnx/vision_encoder_q8.onnx", 126353L),
                        FileSpec("onnx/vision_encoder_q8.onnx_data", 196153344L)
                    )),
                PrecisionInfo("q4f16", "onnx/decoder_q4f16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q4f16.onnx", 278229L),
                        FileSpec("onnx/decoder_q4f16.onnx_data", 1413388288L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("q8f16", "onnx/decoder_q8f16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q8f16.onnx", 273077L),
                        FileSpec("onnx/decoder_q8f16.onnx_data", 2055898112L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("fp16", "onnx/decoder_fp16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_fp16.onnx", 280626L),
                        FileSpec("onnx/decoder_fp16.onnx_data", 2128306176L),
                        FileSpec("onnx/decoder_fp16.onnx_data_1", 1493966848L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("fp32", "onnx/decoder.onnx", "onnx/vision_encoder.onnx",
                    listOf(
                        FileSpec("onnx/decoder.onnx", 284313L),
                        FileSpec("onnx/decoder.onnx_data", 2120695808L),
                        FileSpec("onnx/decoder.onnx_data_1", 2135916544L),
                        FileSpec("onnx/decoder.onnx_data_2", 2140200960L),
                        FileSpec("onnx/decoder.onnx_data_3", 847732736L),
                        FileSpec("onnx/vision_encoder.onnx", 105627L),
                        FileSpec("onnx/vision_encoder.onnx_data", 758042624L)
                    ))
            )
        ),
        ModelInfo(
            id = "d1-2B-ONNX",
            repo = "LiquidAI/d1-2B-ONNX",
            displayName = "d1 2B (Entscheidungen)",
            kind = ModelKind.DECISION,
            desc = "Ein-Pass-Entscheidungen in Millisekunden: Objekt-Boxen (box/boxes), " +
                "Zählen (number), Auswahl, Ja/Nein – über POST /v1/decide. Kein Chat " +
                "(dafür LFM3-VL-2B laden).",
            needsVision = true,
            commonFiles = listOf(
                FileSpec("tokenizer.json", 17908258L),
                FileSpec("tokenizer_config.json", 623L),
                FileSpec("chat_template.jinja", 12762L),
                FileSpec("processor_config.json", 1254L),
                FileSpec("config.json", 2903L)
            ),
            precisions = listOf(
                PrecisionInfo("q4", "onnx/decoder_q4.onnx", "onnx/vision_encoder_q8.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q4.onnx", 358945L),
                        FileSpec("onnx/decoder_q4.onnx_data", 1741806592L),
                        FileSpec("onnx/vision_encoder_q8.onnx", 126353L),
                        FileSpec("onnx/vision_encoder_q8.onnx_data", 196153344L)
                    )),
                PrecisionInfo("q8", "onnx/decoder_q8.onnx", "onnx/vision_encoder_q8.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q8.onnx", 358439L),
                        FileSpec("onnx/decoder_q8.onnx_data", 2131090432L),
                        FileSpec("onnx/vision_encoder_q8.onnx", 126353L),
                        FileSpec("onnx/vision_encoder_q8.onnx_data", 196153344L)
                    )),
                PrecisionInfo("q4f16", "onnx/decoder_q4f16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q4f16.onnx", 274492L),
                        FileSpec("onnx/decoder_q4f16.onnx_data", 1567474688L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("q8f16", "onnx/decoder_q8f16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_q8f16.onnx", 274026L),
                        FileSpec("onnx/decoder_q8f16.onnx_data", 1956758528L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("fp16", "onnx/decoder_fp16.onnx", "onnx/vision_encoder_fp16.onnx",
                    listOf(
                        FileSpec("onnx/decoder_fp16.onnx", 281560L),
                        FileSpec("onnx/decoder_fp16.onnx_data", 2125680640L),
                        FileSpec("onnx/decoder_fp16.onnx_data_1", 1397452800L),
                        FileSpec("onnx/vision_encoder_fp16.onnx", 104940L),
                        FileSpec("onnx/vision_encoder_fp16.onnx_data", 379021312L)
                    )),
                PrecisionInfo("fp32", "onnx/decoder.onnx", "onnx/vision_encoder.onnx",
                    listOf(
                        FileSpec("onnx/decoder.onnx", 285437L),
                        FileSpec("onnx/decoder.onnx_data", 2111160320L),
                        FileSpec("onnx/decoder.onnx_data_1", 2140200960L),
                        FileSpec("onnx/decoder.onnx_data_2", 2115108864L),
                        FileSpec("onnx/decoder.onnx_data_3", 679796736L),
                        FileSpec("onnx/vision_encoder.onnx", 105627L),
                        FileSpec("onnx/vision_encoder.onnx_data", 758042624L)
                    ))
            )
        )
    )
    // ===== GENERATED-CATALOG-END =====

    /** Menschenlesbare Größe. */
    fun humanBytes(b: Long): String {
        if (b <= 0) return "?"
        val gb = b / 1_000_000_000.0
        if (gb >= 1.0) return String.format(java.util.Locale.US, "%.2f GB", gb)
        return String.format(java.util.Locale.US, "%.0f MB", b / 1_000_000.0)
    }
}
