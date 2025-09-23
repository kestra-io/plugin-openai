package io.kestra.plugin.openai;

import io.kestra.core.junit.annotations.KestraTest;
import org.assertj.core.util.Strings;

@KestraTest
public class AbstractOpenAITest {
    private static final String OPENAI_API_KEY = System.getenv("OPENAI_API_KEY");

    protected static boolean canNotBeEnabled() {
        return Strings.isNullOrEmpty(getApiKey());
    }

    protected static String getApiKey() {
        return OPENAI_API_KEY;
    }
}
