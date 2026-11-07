package ai.liquid.lfm3.engine

/**
 * Chat-Template im LFM3/ChatML-Stil:
 *
 *   <|im_start|>system\n{sys}<|im_end|>\n
 *   <|im_start|>user\n{msg}<|im_end|>\n
 *   <|im_start|>assistant\n<think>im_end|>\n   (Historie: leerer Think-Block)
 *   <|im_start|>assistant\n[<think>\n]         (Generierungs-Prompt)
 */
object ChatTemplate {

    const val IM_START = "<|im_start|>"
    const val IM_END = "<|im_end|>\n"

    // Spezielle IDs (LFM3-Vokabular)
    const val TOKEN_IM_END = 124900
    const val TOKEN_STOP_ALT = 124895
    const val TOKEN_THINK_START = 124901
    const val TOKEN_THINK_END = 124902

    val STOP_IDS = intArrayOf(TOKEN_IM_END, TOKEN_STOP_ALT)

    fun system(sys: String): String = "$IM_START system\n$sys$IM_END"

    fun user(content: String): String = "$IM_START user\n$content$IM_END"

    fun assistantHistory(content: String): String =
        "$IM_START assistant\n<think></think>$content$IM_END"

    fun generationPrompt(thinking: Boolean): String =
        "$IM_START assistant\n" + if (thinking) "<think>\n" else ""

    /** Baut den vollständigen Prompt aus System-Prompt und Verlauf (max. maxTurns Nachrichten). */
    fun render(
        systemPrompt: String,
        history: List<Pair<String, String>>, // (role, content) in chronologischer Reihenfolge
        thinking: Boolean,
        maxTurns: Int = 12
    ): String {
        val sb = StringBuilder()
        if (systemPrompt.isNotBlank()) sb.append(system(systemPrompt))
        val trimmed = history.takeLast(maxTurns)
        for ((role, content) in trimmed) {
            when (role) {
                "user" -> sb.append(user(content))
                "assistant" -> sb.append(assistantHistory(content))
                "system" -> sb.append(system(content))
            }
        }
        sb.append(generationPrompt(thinking))
        return sb.toString()
    }
}
