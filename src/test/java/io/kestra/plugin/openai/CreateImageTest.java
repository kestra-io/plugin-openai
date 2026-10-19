package io.kestra.plugin.openai;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIf;
import org.mockito.ArgumentCaptor;

import com.openai.client.OpenAIClient;
import com.openai.models.images.Image;
import com.openai.models.images.ImageGenerateParams;
import com.openai.models.images.ImagesResponse;
import com.openai.services.blocking.ImageService;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;

import jakarta.inject.Inject;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

@KestraTest
class CreateImageTest extends AbstractOpenAITest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void runWithDallEDownloadTrueStoresImage() throws Exception {
        var runContext = runContextFactory.of();
        byte[] imageBytes = "sample image bytes".getBytes(StandardCharsets.UTF_8);
        var response = ImagesResponse.builder()
            .created(1L)
            .data(List.of(Image.builder().b64Json(Base64.getEncoder().encodeToString(imageBytes)).build()))
            .build();
        var stub = taskWithResponse(true, response);

        var output = stub.task().run(runContext);

        assertThat(output.getImages().size(), is(1));
        URI imageUri = output.getImages().getFirst();
        assertThat(runContext.storage().isFileExist(imageUri), is(true));
        try (var storedImage = runContext.storage().getFile(imageUri)) {
            assertArrayEquals(imageBytes, storedImage.readAllBytes());
        }
        var params = ArgumentCaptor.forClass(ImageGenerateParams.class);
        verify(stub.imageService()).generate(params.capture());
        assertThat(params.getValue().responseFormat().orElseThrow(), is(ImageGenerateParams.ResponseFormat.B64_JSON));
    }

    @Test
    void runWithDallEDownloadFalseReturnsRemoteUrl() throws Exception {
        var runContext = runContextFactory.of();
        URI remoteUri = URI.create("https://example.com/generated-image.png");
        var response = ImagesResponse.builder()
            .created(1L)
            .data(List.of(Image.builder().url(remoteUri.toString()).build()))
            .build();
        var stub = taskWithResponse(false, response);

        var output = stub.task().run(runContext);

        assertThat(output.getImages(), is(List.of(remoteUri)));
        var params = ArgumentCaptor.forClass(ImageGenerateParams.class);
        verify(stub.imageService()).generate(params.capture());
        assertThat(params.getValue().responseFormat().isEmpty(), is(true));
    }

    @Test
    void runWithDefaultModelStoresImage() throws Exception {
        var runContext = runContextFactory.of();
        byte[] imageBytes = "default model image".getBytes(StandardCharsets.UTF_8);
        var response = ImagesResponse.builder()
            .created(1L)
            .data(List.of(Image.builder().b64Json(Base64.getEncoder().encodeToString(imageBytes)).build()))
            .build();
        var stub = taskWithResponse(false, response, null);

        var output = stub.task().run(runContext);

        assertThat(output.getImages().size(), is(1));
        assertThat(runContext.storage().isFileExist(output.getImages().getFirst()), is(true));
        var params = ArgumentCaptor.forClass(ImageGenerateParams.class);
        verify(stub.imageService()).generate(params.capture());
        assertThat(params.getValue().responseFormat().isEmpty(), is(true));
    }

    private static StubbedTask taskWithResponse(boolean download, ImagesResponse response) throws Exception {
        return taskWithResponse(download, response, "dall-e-3");
    }

    private static StubbedTask taskWithResponse(boolean download, ImagesResponse response, String model) throws Exception {
        var client = mock(OpenAIClient.class);
        var imageService = mock(ImageService.class);
        doReturn(imageService).when(client).images();
        doReturn(response).when(imageService).generate(any(ImageGenerateParams.class));

        var builder = CreateImage.builder()
            .apiKey(Property.ofValue("test-key"))
            .prompt(Property.ofValue("A funny cat in a black suit"))
            .download(Property.ofValue(download));
        if (model != null) {
            builder.model(Property.ofValue(model));
        }

        var task = spy(builder.build());
        doReturn(client).when(task).openAIClient(any(RunContext.class));
        return new StubbedTask(task, imageService);
    }

    private record StubbedTask(CreateImage task, ImageService imageService) {
    }

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
