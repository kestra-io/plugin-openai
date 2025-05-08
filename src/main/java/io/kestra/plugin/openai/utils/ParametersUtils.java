package io.kestra.plugin.openai.utils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.openai.models.responses.Response;
import com.openai.models.responses.ResponseInputContent;
import com.openai.models.responses.ResponseInputItem;
import com.openai.models.responses.ResponseInputText;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import static io.kestra.core.utils.Rethrow.throwFunction;

public final class ParametersUtils {
    private static final ObjectMapper OBJECT_MAPPER = JacksonMapper.ofJson();

    private ParametersUtils() {
        // prevent instantiation
    }

    public static ResponseInputItem.Message listParameters(RunContext runContext, Object parameters) throws IllegalVariableEvaluationException {
        switch (parameters) {
            case null -> {
                return null;
            }
            case String str -> {
                String rendered = runContext.render(str);
                return ResponseInputItem.Message.builder()
                    .role(ResponseInputItem.Message.Role.USER)
                    .addContent(ResponseInputContent.ofInputText(
                        ResponseInputText.builder()
                            .text(rendered)
                            .build()
                    ))
                    .build();
            }
            case ResponseInputItem.Message message -> {
                return runContext.render(Property.of(message)).as(ResponseInputItem.Message.class).orElseThrow();
            }
            default -> throw new IllegalArgumentException("Invalid rendered type: " + parameters.getClass());
        }

    }
}