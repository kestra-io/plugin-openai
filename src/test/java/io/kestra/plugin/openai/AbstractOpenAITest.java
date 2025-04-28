package io.kestra.plugin.openai;

import io.kestra.core.junit.annotations.KestraTest;
import org.assertj.core.util.Strings;
import org.junit.jupiter.api.Disabled;

@KestraTest
public class AbstractOpenAITest {
    private static final String OPEN_API_KEY = "";

    protected static boolean canNotBeEnabled() {
        return Strings.isNullOrEmpty(getApiKey());
    }

    protected static String getApiKey() {
        return OPEN_API_KEY;
    }
}
