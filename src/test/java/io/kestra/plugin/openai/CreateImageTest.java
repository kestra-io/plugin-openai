package io.kestra.plugin.openai;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIf;

import com.openai.models.images.ImageGenerateParams;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;

import jakarta.inject.Inject;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

@KestraTest
class CreateImageTest extends AbstractOpenAITest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void buildParamsWithDefaultModelDownloadFalseDoesNotSetResponseFormat() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .download(Property.ofValue(Boolean.FALSE))
            .build();

        ImageGenerateParams params = task.buildParams(runContext);

        assertThat(params.responseFormat().isEmpty(), is(true));
    }

    @Test
    void buildParamsWithDefaultModelDownloadTrueDoesNotSetResponseFormat() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .download(Property.ofValue(Boolean.TRUE))
            .build();

        ImageGenerateParams params = task.buildParams(runContext);

        assertThat(params.responseFormat().isEmpty(), is(true));
    }

    @Test
    void buildParamsWithGptImageMiniDownloadTrueDoesNotSetResponseFormat() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .model(Property.ofValue("gpt-image-1-mini"))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .download(Property.ofValue(Boolean.TRUE))
            .build();

        ImageGenerateParams params = task.buildParams(runContext);

        assertThat(params.responseFormat().isEmpty(), is(true));
    }

    @Test
    void buildParamsWithDallEDownloadFalseDoesNotSetResponseFormat() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .model(Property.ofValue("dall-e-3"))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .download(Property.ofValue(Boolean.FALSE))
            .build();

        ImageGenerateParams params = task.buildParams(runContext);

        assertThat(params.responseFormat().isEmpty(), is(true));
    }

    @Test
    void buildParamsWithDallE3DownloadTrueSetsB64Json() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .model(Property.ofValue("dall-e-3"))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .download(Property.ofValue(Boolean.TRUE))
            .build();

        ImageGenerateParams params = task.buildParams(runContext);

        assertThat(params.responseFormat().isPresent(), is(true));
        assertThat(params.responseFormat().get(), is(ImageGenerateParams.ResponseFormat.B64_JSON));
    }

    @Test
    void buildParamsWithDallE2DownloadTrueSetsB64Json() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .model(Property.ofValue("dall-e-2"))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .download(Property.ofValue(Boolean.TRUE))
            .build();

        ImageGenerateParams params = task.buildParams(runContext);

        assertThat(params.responseFormat().isPresent(), is(true));
        assertThat(params.responseFormat().get(), is(ImageGenerateParams.ResponseFormat.B64_JSON));
    }

    @Test
    @DisabledIf(
        value = "canNotBeEnabled",
        disabledReason = "Needs an OpenAI API Key to work"
    )
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

        CreateImage.Output runOutput = skipOnRateLimit(() -> task.run(runContext));

        assertThat(runOutput.getImages().size(), is(1));
    }

    @Test
    @DisabledIf(
        value = "canNotBeEnabled",
        disabledReason = "Needs an OpenAI API Key to work"
    )
    void runPromptB64Json() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .apiKey(Property.ofValue(getApiKey()))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .size(Property.ofValue(CreateImage.SIZE.LARGE))
            .download(Property.ofValue(Boolean.TRUE))
            .n(1)
            .user(Property.ofValue("test-user"))
            .build();

        CreateImage.Output runOutput = skipOnRateLimit(() -> task.run(runContext));

        assertThat(runOutput.getImages().size(), is(1));
    }
}
