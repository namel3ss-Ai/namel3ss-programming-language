package com.namel3ss.intellij

import com.intellij.lang.annotation.AnnotationHolder
import com.intellij.lang.annotation.Annotator
import com.intellij.openapi.editor.DefaultLanguageHighlighterColors
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.psi.PsiElement
import java.util.regex.Pattern

class Namel3ssCommentAnnotator : Annotator {
    private val commentPattern = Pattern.compile("^#\\s\\S.*")

    override fun annotate(element: PsiElement, holder: AnnotationHolder) {
        val text = element.text ?: return
        if (!commentPattern.matcher(text).find()) return

        holder
            .newSilentAnnotation(com.intellij.lang.annotation.HighlightSeverity.INFORMATION)
            .range(element.textRange)
            .textAttributes(COMMENT)
            .create()
    }

    companion object {
        val COMMENT: TextAttributesKey = TextAttributesKey.createTextAttributesKey(
            "NAMELESS_COMMENT",
            DefaultLanguageHighlighterColors.LINE_COMMENT
        )
    }
}
