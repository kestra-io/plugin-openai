package io.kestra.plugin.openai;

import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIf;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;


@KestraTest
@DisabledIf(
    value = "canNotBeEnabled",
    disabledReason = "Needs an OpenAI API Key to work"
)
class CreateImageTest extends AbstractOpenAITest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void runPromptUrl() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .size(Property.ofValue(CreateImage.SIZE.LARGE))
            .n(1)
            .download(Property.ofValue(Boolean.FALSE))
            .user(Property.ofValue("test-user"))
            .build();

        CreateImage.Output runOutput = task.run(runContext);

        assertThat(runOutput.getImages().size(), is(1));
    }

    @Test
    void runPromptB64Json() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .size(Property.ofValue(CreateImage.SIZE.SMALL))
            .download(Property.ofValue(Boolean.TRUE))
            .n(1)
            .user(Property.ofValue("test-user"))
            .build();

        CreateImage.Output runOutput = task.run(runContext);

        assertThat(runOutput.getImages().size(), is(1));
    }
}
